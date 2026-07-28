import os
import time
import json
import torch
import random
import numpy as np
from copy import deepcopy
from utils import *
from config import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import gc
import weakref

# Input Dimension
INPUT_DIM = 1025
# Set up distributed training environment
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seed = 0 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = BATCH_SIZE
patchlizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                          n_embd=HIDDIEN_DIM,
                          n_head=8,
                          max_length=PATCH_LENGTH,
                          max_position_embeddings=PATCH_LENGTH,
                          vocab_size=1)
char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS,
                         n_embd=HIDDIEN_DIM,
                         n_head=8,
                         max_length=PATCH_SIZE + 1,
                         max_position_embeddings=PATCH_SIZE + 1,
                         vocab_size=128)


model = ListenerT5(patch_config, char_config)
model = model.to(device)

# print parameter number
print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def clear_unused_tensors():
    gc.disable()  # Temporarily disable garbage collection
    try:
        # Get the set of tensor ids used by the model
        if hasattr(model, "module"):
            model_tensors = {id(p) for p in model.module.parameters()}
        else:
            model_tensors = {id(p) for p in model.parameters()}
        
        # Get the set of tensor ids used by the optimizer
        optimizer_tensors = {
            id(state) 
            for state_dict in optimizer.state.values() 
            for state in state_dict.values()
            if isinstance(state, torch.Tensor)  # Ensure only tensors are considered
        }

        # List of all CUDA tensors currently in memory
        tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
        
        # Create weak references to avoid interfering with garbage collection
        tensor_refs = [weakref.ref(tensor) for tensor in tensors]

        for tensor_ref in tensor_refs:
            tensor = tensor_ref()  # Dereference the weak reference
            if tensor is not None and id(tensor) not in model_tensors and id(tensor) not in optimizer_tensors:
                # Mark the tensor for deletion
                tensor.detach_()  # Detach from computation graph
                del tensor  # Delete the tensor reference
    except:
        pass

    finally:
        gc.enable()  # Re-enable garbage collection
        gc.collect()  # Force a garbage collection
        torch.cuda.empty_cache()  # Clear the CUDA cache

def collate_batch(batch):
    audio_features, audio_masks = [], []
    output_patches, output_masks = [], []

    for audio_feature, output_patch in batch:
        audio_features.append(audio_feature)
        audio_masks.append(torch.ones(audio_feature.shape[1])) # mask for time_steps
        output_patches.append(output_patch)
        output_masks.append(torch.tensor([1] * output_patch.shape[0]))
    
    # Pad audio features:
    audio_features = torch.nn.utils.rnn.pad_sequence(
        [f.transpose(0, 1) for f in audio_features],
        batch_first=True,
        padding_value=0
    ).transpose(1, 2) 

    audio_masks = torch.nn.utils.rnn.pad_sequence(
        audio_masks, batch_first=True, padding_value=0)
    
    output_patches = torch.nn.utils.rnn.pad_sequence(
        output_patches, batch_first=True, padding_value=0)
    
    output_masks = torch.nn.utils.rnn.pad_sequence(
        output_masks, batch_first=True, padding_value=0
    )

    return (audio_features.to(device), audio_masks.to(device),
            output_patches.to(device), output_masks.to(device))

class ListenerT5DataSet(Dataset):
    def __init__(self, items):
        # Only save path, not loading data
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        # Only load data when needed
        item = self.items[idx]
        
        try:
        # Load audio features
            audio_feature = np.load(resolve_path(item["spectrogram"]))
            if audio_feature.shape[0] != INPUT_DIM:
                raise ValueError(f"Shape mismatch: {audio_feature.shape}")
            
            if audio_feature.shape[1] > PATCH_LENGTH:
                print(f"{item['spectrogram']} has a length of {audio_feature.shape[1]}, repick")
                random_idx = random.randint(0, len(self.items) - 1)
                return self.__getitem__(random_idx)

                
            audio_feature = torch.tensor(audio_feature, dtype=torch.float32)
        
            # Encode the output
            abc_path = resolve_path(item["output"])
            with open(abc_path, 'r', encoding='utf-8') as f:
                abc_notes = f.read()
            output_patch = patchlizer.encode_train(
                abc_notes, 
                add_special_patches=True
            )
            
            ## Truncate and save samples that exceed 3600 patches
            current_len = len(output_patch)
            if current_len > PATCH_LENGTH:
                print("\n" + "!"*50)
                print(f"An overlong sample was encountered during training!")
                print(f"Spectrogram Path: {item['spectrogram']}")
                print(f"Computed patch length: {current_len}")
                print(f"Original ABC length: {len(abc_notes)}")
                print("!"*50 + "\n")
                with open(f"CRASH_SAMPLE_rank{global_rank}.txt", "w") as f:
                    f.write(f"Spectrogram: {item['spectrogram']}\n")
                    f.write(f"ABC: {abc_notes}\n")
                    f.write(f"Encoded: {output_patch}\n")

            if len(output_patch) > PATCH_LENGTH:
                output_patch = output_patch[:PATCH_LENGTH]
            
            output_patch = torch.tensor(output_patch, dtype=torch.long)
            
            return audio_feature, output_patch
        
        except Exception as e:
            print(f"⚠️ Error loading {item['spectrogram']}: {e}. Replacing with random sample.")
            # Strategy: if the current sample is broken, randomly pick another sample instead
            # This avoids interrupting DataLoader batch assembly
            random_idx = random.randint(0, len(self.items) - 1)
            return self.__getitem__(random_idx)

# Call model with a batch of input
def process_one_batch(batch):
    audio_features, audio_masks, output_patches, output_masks = batch
    loss = model(audio_features=audio_features,
                 audio_masks=audio_masks,
                 decoder_patches=output_patches,
                 decoder_masks=output_masks)
    
    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss

# One epoch for training
def train_epoch():
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()

    for batch in tqdm_train_set:
        with autocast():
            loss = process_one_batch(batch)
        if loss is None or torch.isnan(loss).item():
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        tqdm_train_set.set_postfix({str(global_rank) +'_train_loss': total_train_loss / iter_idx})
        iter_idx += 1

        if iter_idx % 1000 == 0:
            clear_unused_tensors()
    
    return total_train_loss / (iter_idx - 1)

# One epoch for evaluation
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm_eval_set:
        with torch.no_grad():
            loss = process_one_batch(batch)
        if loss is None or torch.isnan(loss).item():
            continue
        total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank) +'_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1
    return total_eval_loss / (iter_idx - 1 )



# Train and Eval
if __name__ == "__main__":
    
    if global_rank == 0:
        os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        with open(LOGS_PATH, 'a', encoding='utf-8') as f:
            f.write("\n=== pretrain run started ===\n")
            f.write(f"log_path: {LOGS_PATH}\n")
            f.write(f"time: {time.asctime(time.localtime(time.time()))}\n\n")
    
    train_set = []
    eval_set = []

    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            train_set.append(data)
    with open(VALIDATION_DATA_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            eval_set.append(data)

    batch_size = max(1, min(batch_size, len(train_set), len(eval_set)))

    train_batch_nums = int(len(train_set) / batch_size)
    eval_batch_nums = int(len(eval_set) / batch_size)


    random.shuffle(train_set)
    random.shuffle(eval_set)


    train_set = train_set[:train_batch_nums * batch_size]
    eval_set = eval_set[:eval_batch_nums * batch_size]


    # Create Dataset 
    train_set = ListenerT5DataSet(train_set)
    eval_set = ListenerT5DataSet(eval_set)


    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=local_rank)


    train_set = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler, shuffle=(train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler, shuffle=(eval_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps= 1000)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LOAD_FROM_PRETRAINED and os.path.exists(PRETRAINED_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')

        # For multi-GPU training, load the model state dict into the model.module
        if torch.cuda.device_count() > 1:
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        
        print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['loss']}")
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')
    
    if LOAD_FROM_CHECKPOINT and os.path.exists(WEIGHTS_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

        # Same as above
        if torch.cuda.device_count() > 1:
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        # Update
        pre_epoch = checkpoint.get('epoch', 0)
        best_epoch = checkpoint.get('best_epoch', 0)
        min_eval_loss = checkpoint.get('min_eval_loss', float('inf'))
        print("Successfully Loaded Checkpoint at Epoch %d" % pre_epoch)
        checkpoint = None
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')
    
    for epoch in range(1+pre_epoch, NUM_EPOCHS+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch" + str(epoch) + '-' * 21)
        train_loss = train_epoch()
        eval_loss = eval_epoch()

        
        if global_rank == 0:
            with open(LOGS_PATH, 'a') as f:
                f.write("Epoch " + str(epoch) +"\ntrain_loss: "+ str(train_loss) + "\neval_loss: " + str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")

            checkpoint = {
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'best_epoch': best_epoch,
                'min_eval_loss': min_eval_loss,
            }

            if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
                checkpoint_path = os.path.splitext(WEIGHTS_PATH)[0] + f"_epoch_{epoch}.pth"
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, WEIGHTS_PATH)

            # --- Change 2: update best_epoch and min_eval_loss in real time ---
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                best_epoch = epoch
                best_checkpoint = {
                    'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'min_eval_loss': min_eval_loss,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                }
                torch.save(best_checkpoint, BEST_WEIGHTS_PATH)
            # if eval_loss < min_eval_loss:
            #     best_epoch = epoch
            #     min_eval_loss = eval_loss
            #     checkpoint = {
            #         'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'lr_sched': lr_scheduler.state_dict(),
            #         'epoch': epoch,
            #         'best_epoch': best_epoch,
            #         'min_eval_loss': min_eval_loss
            #         }
            #     torch.save(checkpoint, WEIGHTS_PATH)
            
        if world_size > 1:
            dist.barrier()
    
    if global_rank == 0:
        print("Best Eval Epoch :" + str(best_epoch))
        print("Min Eval Loss :" + str(min_eval_loss))


        

