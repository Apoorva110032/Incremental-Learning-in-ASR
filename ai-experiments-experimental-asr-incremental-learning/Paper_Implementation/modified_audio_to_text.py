import io
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Union

import braceexpand
import numpy as np
import torch
import webdataset as wd
from torch.utils.data import ChainDataset

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging

import pickle

__all__ = [
    'AudioToBPEDataset',
    'TarredAudioToCharDataset',
    'TarredAudioToBPEDataset',
]


def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len, teacher model
    output logits, teacher model self attention layer output maps
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor, FloatTensor, FloatTensor):  A tuple of tuples of
               signal, signal lengths, encoded tokens, encoded tokens length,
               teacher model logits, teacher model self attention layer
               outputs and teacher importance maps.  This collate func 
               assumes the signals are 1d torch tensors (i.e. mono audio), logits, 
               maps and importance maps are 2d torch tensors.
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 8:
        _, audio_lengths, _, tokens_lengths, logits, _, _, sample_ids = packed_batch
    elif len(packed_batch) == 7:
        sample_ids = None
        _, audio_lengths, _, tokens_lengths, logits, _, _ = packed_batch
    else:
        raise ValueError("Expects 7 or 8 tensors in the batch!")
    
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    
    max_tokens_len = max(tokens_lengths).item()
    
    max_seq_len = max([logit.shape[0] for logit in logits])
#     print(f"max_seq_len: {max_seq_len}")
    
    audio_signal, tokens = [], []
    logits_batch = []
    maps_batch = []
    importance_maps_batch = []
    
    for b in batch:
        if len(b) == 9:
            sig, sig_len, tokens_i, tokens_i_len, logits, maps, importance_maps, _ = b
        else:
            sig, sig_len, tokens_i, tokens_i_len, logits, maps, importance_maps = b
        
        if has_audio:
#             print(f"before processing sig: {sig.shape} sig_len: {sig_len} tokens_i: {tokens_i.shape} tokens_i_len: {tokens_i_len} logits: {logits.shape} maps: {maps.shape} encoder_outputs: {encoder_outputs.shape} importance_maps: {importance_maps.shape}")
            
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
#                 print(f"after processing sig: {sig.shape} sig_len: {sig_len}")
            audio_signal.append(sig)
            
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
#             print(f"after processing tokens_i: {tokens_i.shape} tokens_i_len: {tokens_i_len}")
        tokens.append(tokens_i)
        
        seq_len = logits.shape[0]
        if seq_len < max_seq_len:
            pad = (0, 0, 0, max_seq_len - seq_len)
            logits = torch.nn.functional.pad(logits, pad)
            maps = torch.nn.functional.pad(maps, pad)
            importance_maps = torch.nn.functional.pad(importance_maps, pad)
#             print(f"after processing logits: {logits.shape} maps: {maps.shape} encoder_outputs: {encoder_outputs.shape} importance_maps: {importance_maps.shape}")
            
        logits_batch.append(logits)
        maps_batch.append(maps)
        importance_maps_batch.append(importance_maps)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)
#     import logging
#     logging.error(logits_batch)
#     logging.error(maps_batch)
#     logging.error(encoder_outputs_batch)
    logits_batch = torch.stack(logits_batch, dim=0)
    maps_batch = torch.stack(maps_batch, dim=0)
    importance_maps_batch = torch.stack(importance_maps_batch, dim=0)
    
    logits_batch.requires_grad = True
    maps_batch.requires_grad = True
    importance_maps_batch.requires_grad = True
#     print(f"logits_batch: {logits_batch.requires_grad} maps_batch: {maps_batch.requires_grad} encoder_outputs_batch: {encoder_outputs_batch.requires_grad} importance_maps_batch: {importance_maps_batch.requires_grad}")

    if sample_ids is None:
        return audio_signal, audio_lengths, tokens, tokens_lengths, logits_batch, maps_batch, importance_maps_batch
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return audio_signal, audio_lengths, tokens, tokens_lengths, logits_batch, maps_batch, importance_maps_batch, sample_ids

class ASRManifestProcessor:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        bos_id: Id of beginning of sequence symbol to append if not None.
        eos_id: Id of end of sequence symbol to append if not None.
        pad_id: Id of pad symbol. Defaults to 0.
    """

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
    ):
        self.parser = parser

        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
        )

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_id(self, index: int) -> (List[int], int):
        sample = self.collection[index]
        return self.process_text_by_sample(sample)

    def process_text_by_file_id(self, file_id: str) -> (List[int], int):
        manifest_idx = self.collection.mapping[file_id][0]
        sample = self.collection[manifest_idx]
        return self.process_text_by_sample(sample)

    def process_text_by_sample(self, sample: collections.ASRAudioText.OUTPUT_TYPE) -> (List[int], int):
        t, tl = sample.text_tokens, len(sample.text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl

def expand_audio_filepaths(audio_tar_filepaths, shard_strategy: str, world_size: int, global_rank: int):
    valid_shard_strategies = ['scatter', 'replicate']
    if shard_strategy not in valid_shard_strategies:
        raise ValueError(f"`shard_strategy` must be one of {valid_shard_strategies}")

    if isinstance(audio_tar_filepaths, str):
        # Replace '(' and '[' with '{'
        brace_keys_open = ['(', '[', '<', '_OP_']
        for bkey in brace_keys_open:
            if bkey in audio_tar_filepaths:
                audio_tar_filepaths = audio_tar_filepaths.replace(bkey, "{")

        # Replace ')' and ']' with '}'
        brace_keys_close = [')', ']', '>', '_CL_']
        for bkey in brace_keys_close:
            if bkey in audio_tar_filepaths:
                audio_tar_filepaths = audio_tar_filepaths.replace(bkey, "}")

    if isinstance(audio_tar_filepaths, str):
        # Brace expand
        audio_tar_filepaths = list(braceexpand.braceexpand(audio_tar_filepaths))

    # Check for distributed and partition shards accordingly
    if world_size > 1:
        if shard_strategy == 'scatter':
            logging.info("All tarred dataset shards will be scattered evenly across all nodes.")

            if len(audio_tar_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(audio_tar_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size})."
                )

            begin_idx = (len(audio_tar_filepaths) // world_size) * global_rank
            end_idx = begin_idx + len(audio_tar_filepaths) // world_size
            audio_tar_filepaths = audio_tar_filepaths[begin_idx:end_idx]
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )

        elif shard_strategy == 'replicate':
            logging.info("All tarred dataset shards will be replicated across all nodes.")
        else:
            raise ValueError(f"Invalid shard strategy ! Allowed values are : {valid_shard_strategies}")

    return audio_tar_filepaths

class _AudioTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        trim: whether or not to trim silence. Defaults to False
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        pad_id: Id of pad symbol. Defaults to 0
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports. (collate_fn returns this)
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'teacher_logits': NeuralType(('B', 'T', 'D'), LogprobsType()),
            'teacher_feature_map': NeuralType(('B', 'T', 'D'), LogprobsType()),
            'teacher_importance_map': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id
        
        # Loading objects (data) from files (on Disk) to restore their values to memory
        # Loading teacher_logits
        DIR = "/home/DATA2/apoorvaaggarwal/training"
        file = f"{DIR}/teacher_logits.pkl"
        file_obj = open(file, "rb")
        self.teacher_logits = pickle.load(file_obj) # Tensor

        # Loading teacher_feature_map
        file = f"{DIR}/teacher_feature_map.pkl"
        file_obj = open(file, "rb") # read binary
        self.teacher_feature_map = pickle.load(file_obj) # Tensor
        
        file = f"{DIR}/teacher_importance_map.pkl"
        file_obj = open(file, "rb") # read binary
        self.teacher_importance_map = pickle.load(file_obj) # Tensor

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        # A method that specifies how to load a single data instance
        import logging
        sample = self.manifest_processor.collection[index]
        file_path = sample.audio_file
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim, orig_sr=sample.orig_sr
        )
        f, fl = features, torch.tensor(features.shape[0]).long() # Tensor, Tensor

        t, tl = self.manifest_processor.process_text_by_sample(sample=sample) # List, Int
#         logging.error(f"index: {index} file path: {file_path}")
        
        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), torch.tensor(self.teacher_logits[file_path]), torch.tensor(self.teacher_feature_map[file_path]), torch.tensor(self.teacher_importance_map[file_path]), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), torch.tensor(self.teacher_logits[file_path]), torch.tensor(self.teacher_feature_map[file_path]), torch.tensor(self.teacher_importance_map[file_path])
        
        return output

    def __len__(self):
        # A method that returns the total number of data instances
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id)

class AudioToBPEDataset(_AudioTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    In practice, the dataset and manifest used for character encoding and byte pair encoding
    are exactly the same. The only difference lies in how the dataset tokenizes the text in
    the manifest.

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        tokenizer: A subclass of the Tokenizer wrapper found in the common collection,
            nemo.collections.common.tokenizers.TokenizerSpec. ASR Models support a subset of
            all available tokenizers.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        trim: Whether to trim silence segments
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'teacher_logits': NeuralType(('B', 'T', 'D'), LogprobsType()),
            'teacher_feature_map': NeuralType(('B', 'T', 'D'), LogprobsType()),
            'teacher_importance_map': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        return_sample_id: bool = False,
    ):
        if use_start_end_token and hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            return_sample_id=return_sample_id,
        )

class _TarredAudioToTextDataset(IterableDataset):
    """
    A similar Dataset to the AudioToCharDataset/AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToCharDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        parser (callable): A callable which is used to pre-process the text output.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        parser: Callable,
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
    ):
        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.return_sample_id = return_sample_id

        audio_tar_filepaths = expand_audio_filepaths(
            audio_tar_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )

        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=audio_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = (
            self._dataset.rename(audio='wav;ogg;flac', key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .pipe(self._loop_offsets)
            .map(f=self._build_sample)
        )
        
        # Loading objects (data) from files (on Disk) to restore their values to memory
        DIR = "/home/DATA2/apoorvaaggarwal/training"
        # Loading teacher_logits
        file = f"{DIR}/teacher_logits.pkl"
        file_obj = open(file, "rb")
        self.teacher_logits = pickle.load(file_obj) # Tensor

        # Loading teacher_feature_map
        file = f"{DIR}/teacher_feature_map.pkl"
        file_obj = open(file, "rb") # read binary
        self.teacher_feature_map = pickle.load(file_obj) # Tensor

        # Loading teacher_importance_map
        file = f"{DIR}/teacher_importance_map.pkl"
        file_obj = open(file, "rb")  # read binary
        self.teacher_importance_map = pickle.load(file_obj)  # Tensor

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRAudioText already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection

            def __iter__(self):
                return self

            def __next__(self):
                while True:
                    audio_bytes, audio_filename = next(self.iterator)
                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if file_id in self.collection.mapping:
                        return audio_bytes, audio_filename

        return TarredAudioFilter(self.manifest_processor.collection)

    def _loop_offsets(self, iterator):
        """This function is used to iterate through utterances with different offsets for each file.
        """

        class TarredAudioLoopOffsets:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection
                self.current_fn = None
                self.current_bytes = None
                self.offset_id = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.current_fn is None:
                    self.current_bytes, self.current_fn = next(self.iterator)
                    self.offset_id = 0
                else:
                    offset_list = self.collection.mapping[self.current_fn]
                    if len(offset_list) == self.offset_id + 1:
                        self.current_bytes, self.current_fn = next(self.iterator)
                        self.offset_id = 0
                    else:
                        self.offset_id += 1

                return self.current_bytes, self.current_fn, self.offset_id

        return TarredAudioLoopOffsets(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, self.pad_id)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """

        audio_bytes, audio_filename, offset_id = tup

        # Grab manifest entry from self.manifest_preprocessor.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.manifest_processor.collection.mapping[file_id][offset_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream,
            offset=offset,
            duration=manifest_entry.duration,
            trim=self.trim,
            orig_sr=manifest_entry.orig_sr,
        )
        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        # Text features
        t, tl = manifest_entry.text_tokens, len(manifest_entry.text_tokens)

        self.manifest_processor.process_text_by_sample(sample=manifest_entry)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), self.teacher_logits[audio_filename], self.teacher_feature_map[audio_filename], teacher_importance_map[audio_filename], manifest_idx
        else:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), self.teacher_logits[audio_filename], self.teacher_feature_map[audio_filename], teacher_importance_map[audio_filename]

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.manifest_processor.collection)

class TarredAudioToCharDataset(_TarredAudioToTextDataset):
    """
    A similar Dataset to the AudioToCharDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToCharDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        labels: List[str],
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        parser: Optional[str] = 'en',
        pad_id: int = 0,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            shard_strategy=shard_strategy,
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=return_sample_id,
        )

class TarredAudioToBPEDataset(_TarredAudioToTextDataset):
    """
    A similar Dataset to the AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        tokenizer (TokenizerSpec): Either a Word Piece Encoding tokenizer (BERT),
            or a Sentence Piece Encoding tokenizer (BPE). The CTC blank
            symbol is automatically added later for models using ctc.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
    ):
        if use_start_end_token and hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            shard_strategy=shard_strategy,
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=return_sample_id,
        )

class BucketingDataset(IterableDataset):
    """
    A Dataset which wraps another IterableDataset and adopts it for bucketing
    Args:
        dataset (IterableDataset): The IterableDataset to get wrapped
        bucketing_batch_size (int): Number of samples to build a batch
    """

    def __init__(
        self, dataset: IterableDataset, bucketing_batch_size: int,
    ):
        self.wrapped_dataset = dataset
        self.bucketing_batch_size = bucketing_batch_size
        super().__init__()

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch[0], self.wrapped_dataset.pad_id)

    def __iter__(self):
        return BucketingIterator(
            wrapped_iter=self.wrapped_dataset._dataset.__iter__(), bucketing_batch_size=self.bucketing_batch_size
        ).__iter__()

    def __len__(self):
        return int(math.ceil(len(self.wrapped_dataset) / float(self.bucketing_batch_size)))


class BucketingIterator:
    def __init__(self, wrapped_iter, bucketing_batch_size):
        self.wrapped_iter = wrapped_iter
        self.bucketing_batch_size = bucketing_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        batches = []
        for idx in range(self.bucketing_batch_size):
            try:
                sample = next(self.wrapped_iter)
            except StopIteration:
                break
            batches.append(sample)
        if len(batches) == 0:
            raise StopIteration
        return batches


class RandomizedChainDataset(ChainDataset):
    def __init__(self, datasets: Iterable[Dataset], rnd_seed=0) -> None:
        super(RandomizedChainDataset, self).__init__(list(datasets))
        self.rnd_gen = np.random.RandomState(rnd_seed)

    def __iter__(self):
        shuffled_order = self.rnd_gen.permutation(len(self.datasets))
        for dataset_idx in shuffled_order:
            d = self.datasets[dataset_idx]
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x