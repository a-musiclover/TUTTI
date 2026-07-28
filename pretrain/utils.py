import re
import numpy as np
import bisect  # This is for the NotaGen Patchlizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from config import *
from unidecode import unidecode
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from transformers import GPT2LMHeadModel, PreTrainedModel, EncoderDecoderConfig, EncoderDecoderModel

class Patchilizer:
    def __init__(self, stream=PATCH_STREAM):
        self.stream = stream
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0

    def split_bars(self, body_lines):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1] 
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def split_patches(self, abc_text, patch_size=PATCH_SIZE, generate_last=False):
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i : i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def patch2chars(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes
        

    def patchilize_metadata(self, metadata_lines):

        metadata_patches = []
        for line in metadata_lines:
            metadata_patches += self.split_patches(line)

        return metadata_patches
    
    def patchilize_tunebody(self, tunebody_lines, encode_mode='train'):

        tunebody_patches = []
        bars = self.split_bars(tunebody_lines)
        if encode_mode == 'train':
            for bar in bars:
                tunebody_patches += self.split_patches(bar)
        elif encode_mode == 'generate':
            for bar in bars[:-1]:
                tunebody_patches += self.split_patches(bar)
            tunebody_patches += self.split_patches(bars[-1], generate_last=True)
       
        return tunebody_patches

    def encode_train(self, abc_text, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True, cut=True):

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if line.startswith('[V:'):
                tunebody_index = i
                break

        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]

        if self.stream:
            tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                                enumerate(tunebody_lines)]    # [r:n/n]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)

            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if '\n' in patch]
                line_index_for_cut_index = list(range(len(available_cut_indexes)))  
                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index) 
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index] 
                    stream_tunebody_lines = tunebody_lines[line_index : ]
                    
                    stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                    patches = metadata_patches + stream_tunebody_patches
            else:
                patches = metadata_patches + tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches

        if cut: 
            patches = patches[ : patch_length]
        else:   
            pass

        # encode to ids
        id_patches = []
        for patch in patches:
            id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches

    def encode_generate(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True):

        lines = abc_code.split('\n')
        lines = list(filter(None, lines))
    
        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:') or line.startswith('[r:'):
                tunebody_index = i
                break
    
        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]   
    
        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'):
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]
    
        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')
    
        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)

            metadata_patches = [bos_patch] + metadata_patches
    
        patches = metadata_patches + tunebody_patches
        patches = patches[ : patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            if len(patch) < PATCH_SIZE and patch[-1] != chr(self.eos_token_id):
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)
        
        return id_patches

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2chars(patch) for patch in patches)

class PatchLevelEnDecoder(PreTrainedModel):
    """
    A patch-level encoder-decoder model for audio features and abc transcriptions
    """
    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        
        # Not sharing weights
        config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
        self.config = config
        self.base = EncoderDecoderModel(config=self.config)

        self.base.config.pad_token_id = 0
        self.base.config.decoder_start_token_id = 1
    
    def forward(self,
                audio_features: torch.Tensor,
                audio_masks: torch.Tensor,
                decoder_patches: torch.Tensor,
                decoder_masks: torch.Tensor):
        """
        The forward pass of the patch-level encoder-decoder model.
        """

        audio_features = audio_features.to(self.device)
        decoder_patches = torch.nn.functional.one_hot(decoder_patches, num_classes=128).float()
        decoder_patches = decoder_patches.reshape(len(decoder_patches), -1, PATCH_SIZE * 128)
        decoder_patches = self.patch_embedding(decoder_patches.to(self.device))

        if audio_masks == None or decoder_masks == None:
            return self.base(inputs_embeds=audio_features,
                             decoder_inputs_embeds=decoder_patches,
                             output_hidden_states=True)["decoder_hidden_states"][-1]
        else:
            return self.base(inputs_embeds=audio_features,
                             attention_mask=audio_masks,
                             decoder_inputs_embeds=decoder_patches,
                             decoder_attention_mask=decoder_masks,
                             output_hidden_states=True)["decoder_hidden_states"][-1]

class CharLevelDecoder(PreTrainedModel):
    """
    A Char-level Decoder model for abc transcriptions
    """
    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        self.base = GPT2LMHeadModel(config)

    def forward(self, encoded_patches: torch.Tensor, target_patches: torch.Tensor):
        """
        The forward pass of the char-level decoder model.
        """
        target_patches = torch.cat((torch.ones_like(target_patches[:,0:1])*self.bos_token_id, target_patches), dim=1)
        
        # preparing the labels for model training
        target_masks = target_patches == self.pad_token_id
        labels = target_patches.clone().masked_fill(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill(labels == -100, 0)

        # select patches
        if PATCH_SAMPLING_BATCH_SIZE !=0 and PATCH_SAMPLING_BATCH_SIZE<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:PATCH_SAMPLING_BATCH_SIZE])

            target_patches = target_patches[selected_indices, :]
            target_masks = target_masks[selected_indices, :]
            labels = labels[selected_indices, :]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1)

        return self.base(inputs_embeds=inputs_embeds,
                         attention_mask=target_masks,
                         labels=labels)

    def generate(self, encoded_patches: torch.Tensor, tokens: torch.Tensor):
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        """ 
        encoded_patches = encoded_patches.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patches with the input embeddings
        tokens = torch.cat((encoded_patches, tokens[:, 1:, :]), dim=1)

        # Get output from the model
        outputs = self.base(inputs_embeds=tokens)

        # Get probabilities of the next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs


class ListenerT5(PreTrainedModel):
    """
    ListenerT5 is a Encoder-Decoder model for audio features and abc transcriptions
    It uses T5 as the backbone model and utilizes the hierarchical structure for Bar and Char-level decoder
    """

    def __init__(self, encoder_config, decoder_config, audio_config=None):
        super().__init__(encoder_config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2


        self.input_projection = nn.Linear(1025, 512) # Add a linear projection layer 
        self.patch_level_endecoder = PatchLevelEnDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)
    
    def forward(self,
                audio_features: torch.Tensor,
                audio_masks: torch.Tensor,
                decoder_patches: torch.Tensor,
                decoder_masks: torch.Tensor):
        """
        The forward pass of the ListenerT5 model.
        """
        decoder_patches = decoder_patches.reshape(len(decoder_patches), -1, PATCH_SIZE)
        if audio_features.dim() == 2:
            encoded_audio = audio_features.transpose(0, 1).unsqueeze(0)
            # print(f"The encoded_audio has a dimension of {encoded_audio.size()}")
        else:
            encoded_audio = audio_features.transpose(1, 2)
            # print(f"The encoded_audio has a dimension of {encoded_audio.size()}")
        encoded_audio = encoded_audio.to(self.device)
        encoded_audio = self.input_projection(encoded_audio)

        encoded_patches = self.patch_level_endecoder(
            audio_features=encoded_audio,
            audio_masks=audio_masks,
            decoder_patches=decoder_patches,
            decoder_masks=decoder_masks
        )

        left_shift_masks = decoder_masks * (decoder_masks.flip(1).cumsum(1).flip(1) > 1)
        decoder_masks[:, 0] = 0

        encoded_patches = encoded_patches[left_shift_masks == 1]
        decoder_patches = decoder_patches[decoder_masks == 1]

        return self.char_level_decoder(encoded_patches,
                                       decoder_patches)["loss"]
    
    def generate(self,
                 audio_features: torch.Tensor,
                 decoder_patches: torch.Tensor,
                 top_p: float=1,
                 top_k: int=0,
                 temperature: float=1,
                 seed: int=None):
        """
        The generate function for generating abc transcription based on the audio features.
        """

        if decoder_patches.shape[-1] % PATCH_SIZE != 0:
            tokens = decoder_patches[:,:,-(decoder_patches.shape[-1]%PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            decoder_patches = decoder_patches[:,:,:-(decoder_patches.shape[-1]%PATCH_SIZE)]
        else:
            tokens =  torch.tensor([self.bos_token_id], device=self.device)

        # encoded_audio = self.audio_encoder(audio_features)
        encoded_audio = audio_features.transpose(0, 1).unsqueeze(0)
        # Linear projection
        encoded_audio = encoded_audio.to(self.device)
        encoded_audio = self.input_projection(encoded_audio)
        
        decoder_patches = decoder_patches.reshape(len(decoder_patches), -1, PATCH_SIZE) # [bs, seq, patch_size]
        encoded_patches = self.patch_level_endecoder(audio_features=encoded_audio,
                                                     audio_masks=None,
                                                     decoder_patches=decoder_patches,
                                                     decoder_masks=None)
        generated_patch = []  

        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()  # [128]
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True) # [128]
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True) # [128]
            token = temperature_sampling(prob, temperature=temperature) # int
            char = chr(token)
            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:# or token == self.eos_token_id:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch

        
