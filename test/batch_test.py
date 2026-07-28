import os
import logging
from tqdm import tqdm
import json
import time
import torch
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = TEST_DIR.parent / "finetune"
TEST_OUTPUT_ROOT = TEST_DIR / "outputs"
LOG_DIR = TEST_OUTPUT_ROOT / "logs"
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

from utils import *
import re
from config import *
from transformers import GPT2Config
import argparse
from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re, Barline_regexPattern
from abctoolkit.transpose import Note_list, Pitch_sign_list
from abctoolkit.duration import calculate_bartext_duration

# Batch inference
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
import queue


# Hyperparameter settings
# TOP_P = 0.8, TOP_K = 8, TEMP = 1.2 matches the legacy model setup
patchilizer = Patchilizer()
TOP_P = 0.3
TOP_K = 1
TEMPERATURE = 0.4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOG_DIR.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(str(LOG_DIR / 'batch_test.log'), mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# Single-file debug mode
def debug_single_file(target_spectrogram_path, output_dir, gpu_id=0):
    """
    Single-thread debug mode: process only one target file.
    """
    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f"Requested GPU {gpu_id}, but only {torch.cuda.device_count()} visible CUDA device(s) are available.")

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print("--- DEBUG MODE ---")
    print(f"Target File: {target_spectrogram_path}")
    print(f"Device: {device}")

    print("Loading model...")
    patchilizer = Patchilizer()

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

    checkpoint = torch.load(TEST_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    os.makedirs(output_dir, exist_ok=True)

    try:
        print("Starting inference...")
        inference_patch_worker(target_spectrogram_path, output_dir, model, patchilizer, device)

        print("Success! Inference finished.")

        output_filename = os.path.splitext(os.path.basename(target_spectrogram_path))[0].replace('_spectrogram', '_reduced') + ".abc"
        print(f"Output saved to: {os.path.join(output_dir, output_filename)}")

    except Exception:
        print("\n!!!!!!!!!! ERROR !!!!!!!!!!")
        import traceback
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# Batch inference
def worker_process(gpu_id, task_queue, result_queue, worker_id):
    """Independent process: load its own model and handle tasks."""
    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f"Requested GPU {gpu_id}, but only {torch.cuda.device_count()} visible CUDA device(s) are available.")

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    logger.info(f"Worker {worker_id} is loading the model on GPU {gpu_id}...")

    patchilizer = Patchilizer()
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
    checkpoint = torch.load(TEST_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    logger.info(f"Worker {worker_id} ready! Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break

            audio_path, output_dir = task
            output_filename = os.path.splitext(os.path.basename(audio_path))[0].replace('_spectrogram', '_reduced') + ".abc"
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                logger.info(f"Worker {worker_id} skipped existing file: {output_filename}")
                result_queue.put((worker_id, audio_path, 'skipped'))
                continue

            resolved_audio_path = resolve_path(audio_path)
            if not os.path.exists(resolved_audio_path):
                logger.error(f"Worker {worker_id} input file not found: {resolved_audio_path}")
                result_queue.put((worker_id, audio_path, 'input_not_found'))
                continue

            try:
                inference_patch_worker(resolved_audio_path, output_dir, model, patchilizer, device)
                result_queue.put((worker_id, audio_path, 'success'))
            except Exception as e:
                logger.error(f"Worker {worker_id} failed to process {audio_path}: {e}")
                result_queue.put((worker_id, audio_path, f'failed: {e}'))

        except queue.Empty:
            continue
    logger.info(f"Worker {worker_id} completed all tasks")


def inference_patch_worker(audio_path, output_dir, model, patchilizer, device):
    """Inference function that accepts the model and device as arguments."""

    audio_feature = torch.tensor(np.load(resolve_path(audio_path)), device=device)
    if audio_feature.shape[1] > 2048:
        logger.warning(f"File {audio_path} exceeds the length limit and was skipped")
        return

    bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]
    byte_list = []
    start_time = time.time()

    input_patches = torch.tensor([bos_patch], device=device).reshape(1, -1)

    failure_flag = False
    end_flag = False
    cut_index = None
    tunebody_flag = False

    while True:
        predicted_patch = model.generate(audio_features=audio_feature,
                                         decoder_patches=input_patches.unsqueeze(0),
                                         top_k=TOP_K,
                                         top_p=TOP_P,
                                         temperature=TEMPERATURE)

        if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):
            tunebody_flag = True
            r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
            temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
            predicted_patch = model.generate(audio_features=audio_feature,
                                             decoder_patches=temp_input_patches.unsqueeze(0),
                                             top_k=TOP_K,
                                             top_p=TOP_P,
                                             temperature=TEMPERATURE)
            predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch

        if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
            end_flag = True
            break

        next_patch = patchilizer.decode([predicted_patch])
        for char in next_patch:
            byte_list.append(char)

        patch_end_flag = False
        for j in range(len(predicted_patch)):
            if patch_end_flag:
                predicted_patch[j] = patchilizer.special_token_id
            if predicted_patch[j] == patchilizer.eos_token_id:
                patch_end_flag = True

        predicted_patch = torch.tensor([predicted_patch], device=device)
        input_patches = torch.cat([input_patches, predicted_patch], dim=1)

        if len(byte_list) > 102400:
            failure_flag = True
            break
        if time.time() - start_time > 20 * 60:
            failure_flag = True
            break

        if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
            abc_code = ''.join(byte_list)
            abc_lines = abc_code.split('\n')
            tunebody_index = None
            for i, line in enumerate(abc_lines):
                if line.startswith('[r:') or line.startswith('[V:'):
                    tunebody_index = i
                    break
            if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                break

            metadata_lines = abc_lines[:tunebody_index]
            tunebody_lines = abc_lines[tunebody_index:]
            metadata_lines = [line + '\n' for line in metadata_lines]
            if not abc_code.endswith('\n'):
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]

            if cut_index is None:
                cut_index = len(tunebody_lines) // 2

            abc_code_slice = ''.join(metadata_lines + tunebody_lines[-cut_index:])
            input_patches = patchilizer.encode_generate(abc_code_slice)
            input_patches = [item for sublist in input_patches for item in sublist]
            input_patches = torch.tensor([input_patches], device=device)
            input_patches = input_patches.reshape(1, -1)

    if not failure_flag:
        abc_text = ''.join(byte_list)
        filename = os.path.splitext(os.path.basename(audio_path))[0].replace('_spectrogram', '_reduced') + ".abc"
        unreduced_output_path = os.path.join(output_dir, filename)

        abc_lines = abc_text.split('\n')
        abc_lines = list(filter(None, abc_lines))
        abc_lines = [line + '\n' for line in abc_lines]
        try:
            abc_lines = rest_unreduce(abc_lines)
            with open(unreduced_output_path, 'w') as file:
                file.writelines(abc_lines)
        except Exception as e:
            logger.error(f"Failed to convert file {os.path.splitext(os.path.basename(audio_path))[0]}: {e}")


def parallel_inference(test_data_path, output_dir, gpu_id=0, num_workers=8):
    """Main entry point for parallel inference."""
    os.makedirs(output_dir, exist_ok=True)

    task_queue = Queue(maxsize=num_workers * 2)
    result_queue = Queue()

    processes = []
    for i in range(num_workers):
        p = Process(target=worker_process,
                   args=(gpu_id, task_queue, result_queue, i))
        p.start()
        processes.append(p)
        logger.info(f"Started process {i}")

    tasks = []
    with open(test_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            tasks.append((data['spectrogram'], output_dir))

    logger.info(f"Total {len(tasks)} tasks assigned to {num_workers} processes")

    for task in tasks:
        task_queue.put(task)

    for _ in range(num_workers):
        task_queue.put(None)

    results = []
    with tqdm(total=len(tasks), desc="Overall progress") as pbar:
        for _ in range(len(tasks)):
            result = result_queue.get()
            results.append(result)
            pbar.update(1)
            worker_id, audio_path, status = result
            if status == 'success':
                logger.info(f"Worker {worker_id} finished: {os.path.basename(audio_path)}")

    for p in processes:
        p.join()

    success_count = sum(1 for r in results if r[2] == 'success')
    logger.info(f"Done! Success: {success_count}/{len(tasks)}")

    return results


def rest_unreduce(abc_lines):

    tunebody_index = None
    for i in range(len(abc_lines)):
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[: tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    existed_voices = []
    for line in metadata_lines:
        if line.startswith('%%score'):
            for round_bracket_match in re.findall(r'\((.*?)\)', line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith('V:'):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])
    z_symbol_list = []
    x_symbol_list = []
    for voice_group in voice_group_list:
        z_symbol_list.append('V:' + voice_group[0])
        for j in range(1, len(voice_group)):
            x_symbol_list.append('V:' + voice_group[j])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''
        ref_dur = 1
        right_barline = ''

        line = re.sub(r'^\[r:[^\]]*\]', '', line)

        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)

        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            line_bar_dict[key] = value

        dur_dict = {}
        for symbol, bartext in line_bar_dict.items():
            right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext = bartext[:-len(right_barline)]
            try:
                bar_dur = calculate_bartext_duration(bartext)
            except:
                bar_dur = None
            if bar_dur is not None:
                if bar_dur not in dur_dict.keys():
                    dur_dict[bar_dur] = 1
                else:
                    dur_dict[bar_dur] += 1

        try:
            ref_dur = max(dur_dict, key=dur_dict.get)
        except:
            pass

        if i == 0:
            prefix_left_barline = line.split('[V:')[0]
        else:
            prefix_left_barline = ''

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
            else:
                if symbol in z_symbol_list:
                    symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + right_barline
                elif symbol in x_symbol_list:
                    symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + right_barline
            unreduced_line += '[' + symbol + ']' + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + '\n')

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


if __name__ == '__main__':
    # Keep a local debug entry point here; for production runs, use test/run_test_pipeline.sh
    debug_single_file(
        target_spectrogram_path='/23A052/ListenerT5/saxphone_data/real_a2s_sax_dataset/spectrogram/tenor/AbBebopMajorScale.npyp',
        output_dir=TEST_OUTPUT_DIR,
        gpu_id=0
    )