import torch
import torch.nn as nn
import os
import tempfile
import json
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch
import pandas as pd


from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType, AcousticEncodedRepresentation
from nemo.utils import logging, model_utils
from nemo.utils import logging
import modified_dataset

from overwritten_encoder import OverwrittenEncoder
from total_loss import TotalLoss


__all__ = ['ModifiedModel', 'ModifiedTeacherModel']


class ModifiedModel(EncDecCTCModelBPE, ASRBPEMixin):
    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.encoder = OverwrittenEncoder(feat_in=80, feat_out=-1, n_layers=18, d_model=512, subsampling='striding',
          subsampling_factor=4, subsampling_conv_channels=-1, ff_expansion_factor=4, self_attention_model='rel_pos', n_heads=8, att_context_size=[-1, -1],
          xscaling=True, untie_biases=True, pos_emb_max_len=5000, conv_kernel_size=31, dropout=0.1, dropout_emb=0.0, dropout_att=0.1)

        self.loss = TotalLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            # hyper parameters
            temperature=5, gamma=100, beta=0.01
        )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = modified_dataset.get_dali_bpe_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = modified_dataset.get_tarred_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = modified_dataset.get_bpe_dataset(
                config=config, tokenizer=self.tokenizer, augmentor=augmentor
            )
        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def transcribe(
          self,
          paths2audio_files: List[str],
          batch_size: int = 2,
          logprobs: bool = False,
          return_hypotheses: bool = False,
          return_self_attention_outputs: bool = False,
          num_workers: int = 0,
      ) -> List[str]:
        """
          Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

          Args:
              paths2audio_files: (a list) of paths to audio files. \
                  Recommended length per file is between 5 and 25 seconds. \
                  But it is possible to pass a few hours long file if enough GPU memory is available.
              batch_size: (int) batch size to use during inference.
                  Bigger will result in better throughput performance but would use more memory.
              logprobs: (bool) pass True to get log probabilities instead of transcripts.
              return_hypotheses: (bool) Either return hypotheses or text
                  With hypotheses can do some postprocessing like getting timestamp or rescoring
              return_self_attention_outputs: (bool) pass True to get outputs of self attention layer instead of transcripts or log probabilities.
              num_workers: (int) number of workers for DataLoader

          Returns:
              A list of transcriptions (or raw log probabilities (tensor) if logprobs is True,
              or outputs of self attention layer (tensor) if return_self_attention_outputs
              is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if logprobs and any([return_hypotheses, return_self_attention_outputs]):
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` or `return_self_attention_outputs` can be True at any given time."
                "Returned hypotheses will contain the logprobs or the self attention layer outputs."
            )
        if return_hypotheses and any([logprobs, return_self_attention_outputs]):
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` or `return_self_attention_outputs` can be True at any given time."
                "Returned hypotheses will contain the logprobs or the self attention layer outputs."
            )
        if return_self_attention_outputs and any([return_hypotheses, logprobs]):
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` or `return_self_attention_outputs` can be True at any given time."
                "Returned hypotheses will contain the logprobs or the self attention layer outputs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                }

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    logits, logits_len, greedy_predictions, self_attention_outputs, importance_map = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            # lg = logits[idx][: logits_len[idx]]
                            # hypotheses.append(lg.cpu().numpy())
                            hypotheses.append(logits[idx])
                        # hypotheses += logits
                    elif return_self_attention_outputs:
                        # dump self attention layer outputs per file
                        for idx in range(logits.shape[0]):
                            # sal = self_attention_outputs[idx][: logits_len[idx]]
                            hypotheses.append(self_attention_outputs[idx])
                        # hypotheses += self_attention_outputs
                    else:
                        current_hypotheses = self._wer.ctc_decoder_predictions_tensor(
                            greedy_predictions, predictions_len=logits_len, return_hypotheses=return_hypotheses,
                        )

                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                        hypotheses += current_hypotheses

                    del greedy_predictions
                    del logits
                    del self_attention_outputs
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)
        return hypotheses

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
            "self_attention_outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "importance_map": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
          Forward pass of the model.

          Args:
              input_signal: Tensor that represents a batch of raw audio signals,
                  of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                  `self.sample_rate` number of floating point values.
              input_signal_length: Vector of length B, that contains the individual lengths of the audio
                  sequences.
              processed_signal: Tensor that represents a batch of processed audio signals,
                  of shape (B, D, T) that has undergone processing via some DALI preprocessor.
              processed_signal_length: Vector of length B, that contains the individual lengths of the
                  processed audio sequences.

          Returns:
              A tuple of 4 elements -
              1) The log probabilities tensor of shape [B, T, D].
              2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
              3) The greedy token predictions of the model of shape [B, T] (via argmax)
              4) The outputs of self attention layer, of shape [B, T, D]
              5) The importance map - acoustic representations of shape [B, D, T]
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len, self_attention_outputs = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        
        log_probs = self.decoder(encoder_output=encoded).requires_grad_()
        student_gpp = torch.prod(torch.max(log_probs, 2).values, 1).requires_grad_()
        
#         print(f"student_encoder_outputs: {encoded.requires_grad} self_attention_outputs: {self_attention_outputs.requires_grad} log_probs: {log_probs.requires_grad} student_gpp: {student_gpp.requires_grad}")
        
        importance_map = torch.autograd.grad(student_gpp, self_attention_outputs, grad_outputs=torch.ones_like(student_gpp), retain_graph=True)[0]
        
#         print(f"importance_map: {importance_map} batch_length: {len(importance_map)} length: {len(importance_map[0])} dim: {len(importance_map[0][0])}")
        
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions, self_attention_outputs, importance_map

    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len, teacher_logits, teacher_feature_map, teacher_importance_map = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions, self_attention_outputs, student_importance_map = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions, self_attention_outputs, student_importance_map = self.forward(input_signal=signal, input_signal_length=signal_len)
            
        max_seq_len = max([teacher_logit.shape[0] for teacher_logit in teacher_logits])
#         print(f"max_seq_len: {max_seq_len}")
        
        logits_batch = []
        maps_batch = []
        importance_maps_batch = []
        
        for idx in range(log_probs.shape[0]):
#             print(f"index: {idx}")
#             print(f"signal: {signal[idx].shape} signal_len: {signal_len[idx]} transcript: {transcript[idx].shape} transcript_len: {transcript_len[idx]}")
#             print(f"teacher_logits: {teacher_logits[idx].shape} student_logits: {log_probs[idx].shape} teacher_feature_map: {teacher_feature_map[idx].shape} student_feature_map: {self_attention_outputs[idx].shape} teacher_importance_map: {teacher_importance_map[idx].shape} student_importance_map: {student_importance_map[idx].shape}")
            seq_len = log_probs[idx].shape[0]
            if seq_len < max_seq_len:
                pad = (0, 0, 0, max_seq_len - seq_len)
                logits = torch.nn.functional.pad(log_probs[idx], pad)
                maps = torch.nn.functional.pad(self_attention_outputs[idx], pad)
                importance_maps = torch.nn.functional.pad(student_importance_map[idx], pad)
#                 print(f"after processing student_logit: {log_probs[idx].shape} student_feature_map: {self_attention_outputs[idx].shape} student_importance_map: {student_importance_map[idx].shape}")
                logits_batch.append(logits)
                maps_batch.append(maps)
                importance_maps_batch.append(importance_maps)
            
            else:
                logits_batch.append(log_probs[idx])
                maps_batch.append(self_attention_outputs[idx])
                importance_maps_batch.append(student_importance_map[idx])
        
        logits_batch = torch.stack(logits_batch, dim=0)
        maps_batch = torch.stack(maps_batch, dim=0)
        importance_maps_batch = torch.stack(importance_maps_batch, dim=0)

        # Accessing parameters' values
        log_probs = logits_batch
        targets = transcript
        input_lengths = encoded_len
        target_lengths = transcript_len
        student_feature_map = maps_batch
        student_importance_map=importance_maps_batch
        
        loss_value = self.loss(
            student_logits=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            teacher_logits=teacher_logits,
            teacher_feature_map=teacher_feature_map,
            student_feature_map=student_feature_map,
            teacher_importance_map=teacher_importance_map,
            student_importance_map=student_importance_map
        )
        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1
            
        if batch_nb % 10 == 1:
            torch.cuda.empty_cache()

        if (batch_nb + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=predictions,
                targets=transcript,
                target_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, teacher_logits, teacher_feature_map, teacher_importance_map, sample_id = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions, self_attention_outputs, student_importance_map = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions, self_attention_outputs, student_importance_map = self.forward(input_signal=signal, input_signal_length=signal_len)

        transcribed_texts = self._wer.ctc_decoder_predictions_tensor(
            predictions=predictions, predictions_len=encoded_len, return_hypotheses=False,
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, transcribed_texts))
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, teacher_logits, teacher_feature_map, teacher_importance_map = batch
        with torch.enable_grad():
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                log_probs, encoded_len, predictions, self_attention_outputs, student_importance_map = self.forward(
                    processed_signal=signal, processed_signal_length=signal_len
                )
            else:
                log_probs, encoded_len, predictions, self_attention_outputs, student_importance_map = self.forward(input_signal=signal, input_signal_length=signal_len)

        max_seq_len = max([teacher_logit.shape[0] for teacher_logit in teacher_logits])
#         print(f"max_seq_len: {max_seq_len}")
        
        logits_batch = []
        maps_batch = []
        importance_maps_batch = []
        
        for idx in range(log_probs.shape[0]):
#             print(f"index: {idx}")
#             print(f"signal: {signal[idx].shape} signal_len: {signal_len[idx]} transcript: {transcript[idx].shape} transcript_len: {transcript_len[idx]}")
#             print(f"teacher_logits: {teacher_logits[idx].shape} student_logits: {log_probs[idx].shape} teacher_feature_map: {teacher_feature_map[idx].shape} student_feature_map: {self_attention_outputs[idx].shape} teacher_importance_map: {teacher_importance_map[idx].shape} student_importance_map: {student_importance_map[idx].shape}")
            seq_len = log_probs[idx].shape[0]
            if seq_len < max_seq_len:
                pad = (0, 0, 0, max_seq_len - seq_len)
                logits = torch.nn.functional.pad(log_probs[idx], pad)
                maps = torch.nn.functional.pad(self_attention_outputs[idx], pad)
                importance_maps = torch.nn.functional.pad(student_importance_map[idx], pad)
#                 print(f"after processing student_logit: {log_probs[idx].shape} student_feature_map: {self_attention_outputs[idx].shape} student_importance_map: {student_importance_map[idx].shape}")
                logits_batch.append(logits)
                maps_batch.append(maps)
                importance_maps_batch.append(importance_maps)
            
            else:
                logits_batch.append(log_probs[idx])
                maps_batch.append(self_attention_outputs[idx])
                importance_maps_batch.append(student_importance_map[idx])
        
        logits_batch = torch.stack(logits_batch, dim=0)
        maps_batch = torch.stack(maps_batch, dim=0)
        importance_maps_batch = torch.stack(importance_maps_batch, dim=0)

        # Accessing parameters' values
        log_probs = logits_batch
        targets = transcript
        input_lengths = encoded_len
        target_lengths = transcript_len
        student_feature_map = maps_batch
        student_importance_map=importance_maps_batch

        loss_value = self.loss(
            student_logits=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            teacher_logits=teacher_logits,
            teacher_feature_map=teacher_feature_map,
            student_feature_map=student_feature_map,
            teacher_importance_map=teacher_importance_map,
            student_importance_map=student_importance_map
        )
        self._wer.update(
            predictions=predictions, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()
        del log_probs, targets, input_lengths, target_lengths, teacher_logits, teacher_feature_map, student_feature_map, teacher_importance_map, student_importance_map, predictions, transcript, transcript_len, encoded_len
        
        self.log("val_wer:", wer)
        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
        }


# MODIFIED TEACHER MODEL

class ModifiedTeacherModel(EncDecCTCModelBPE, ASRBPEMixin):
    
    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.encoder = OverwrittenEncoder(feat_in=80, feat_out=-1, n_layers=18, d_model=512, subsampling='striding',
          subsampling_factor=4, subsampling_conv_channels=-1, ff_expansion_factor=4, self_attention_model='rel_pos', n_heads=8, att_context_size=[-1, -1],
          xscaling=True, untie_biases=True, pos_emb_max_len=5000, conv_kernel_size=31, dropout=0.1, dropout_emb=0.0, dropout_att=0.1)

    def transcribe(
          self,
          paths2audio_files: List[str],
          batch_size: int = 2,
          logprobs: bool = False,
          return_hypotheses: bool = False,
          return_self_attention_outputs: bool = False,
          return_importance_map: bool = False,
          num_workers: int = 0,
      ) -> List[str]:
        """
          Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

          Args:
              paths2audio_files: (a list) of paths to audio files. \
                  Recommended length per file is between 5 and 25 seconds. \
                  But it is possible to pass a few hours long file if enough GPU memory is available.
              batch_size: (int) batch size to use during inference.
                  Bigger will result in better throughput performance but would use more memory.
              logprobs: (bool) pass True to get log probabilities instead of transcripts.
              return_hypotheses: (bool) Either return hypotheses or text
                  With hypotheses can do some postprocessing like getting timestamp or rescoring
              return_self_attention_outputs: (bool) pass True to get outputs of self attention layer instead of transcripts or log probabilities.
              return_importance_map: (bool) pass True to get importance map instead of transcripts or log probabilities or self attention layer outputs.
              num_workers: (int) number of workers for DataLoader

          Returns:
              A list of transcriptions (or raw log probabilities (tensor) if logprobs is True,
              or outputs of self attention layer (tensor) if return_self_attention_outputs
              is True) or importance map (tensor) if return_importance_map
              is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if logprobs and any([return_hypotheses, return_self_attention_outputs, return_importance_map]):
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` or `return_self_attention_outputs` or `return_importance_map` can be True at any given time."
                "Returned hypotheses will contain the logprobs or the self attention layer outputs or the importance maps."
            )
        if return_hypotheses and any([logprobs, return_self_attention_outputs, return_importance_map]):
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` or `return_self_attention_outputs` or `return_importance_map` can be True at any given time."
                "Returned hypotheses will contain the logprobs or the self attention layer outputs or the importance maps."
            )
        if return_self_attention_outputs and any([return_hypotheses, logprobs, return_importance_map]):
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` or `return_self_attention_outputs` or `return_importance_map` can be True at any given time."
                "Returned hypotheses will contain the logprobs or the self attention layer outputs or the importance maps."
            )
        if return_importance_map and any([return_hypotheses, logprobs, return_self_attention_outputs]):
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` or `return_self_attention_outputs` or `return_importance_map` can be True at any given time."
                "Returned hypotheses will contain the logprobs or the self attention layer outputs or the importance maps."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                }
                

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    logits, logits_len, greedy_predictions, self_attention_outputs = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
#                     torch.cuda.empty_cache()
#                     print(f"logits: {logits} length: {len(logits)} dim: {len(logits[0])}")
#                     print(f"self_attention_outputs: {self_attention_outputs} length: {len(self_attention_outputs)} dim: {len(self_attention_outputs[0])}")
#                     print(f"encoder_outputs: {encoder_outputs} length: {len(encoder_outputs)} dim: {len(encoder_outputs[0])}")
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            # lg = logits[idx][: logits_len[idx]]
                            # hypotheses.append(lg.cpu().numpy())
                            hypotheses.append(logits[idx].cpu().detach().numpy())
                        # hypotheses += logits
                        
                    elif return_self_attention_outputs:
                        # dump self attention layer outputs per file
                        for idx in range(logits.shape[0]):
                            # sal = self_attention_outputs[idx][: logits_len[idx]]
                            hypotheses.append(self_attention_outputs[idx].cpu().detach().numpy())
                        # hypotheses += self_attention_outputs
                        
                    elif return_importance_map:
                        teacher_gpp = torch.prod(torch.max(logits, 2).values, 1).requires_grad_()
        
#                         print(f"teacher_encoder_outputs: {encoder_outputs.requires_grad} teacher_self_attention_outputs: {self_attention_outputs.requires_grad} teacher_logits: {logits.requires_grad} teacher_gpp: {teacher_gpp.requires_grad}")

                        importance_map = torch.autograd.grad(teacher_gpp, self_attention_outputs, grad_outputs=torch.ones_like(teacher_gpp), retain_graph=True)[0]

#                         print(f"teacher_importance_map: {importance_map} batch_length: {len(importance_map)} length: {len(importance_map[0])} dim: {len(importance_map[0][0])}")
                       
                        for idx in range(logits.shape[0]):
                            # sal = importance_map[idx][: logits_len[idx]]
                            hypotheses.append(importance_map[idx].cpu().detach().numpy())
                        # hypotheses += importance_map
                        
                    else:
                        current_hypotheses = self._wer.ctc_decoder_predictions_tensor(
                            greedy_predictions, predictions_len=logits_len, return_hypotheses=return_hypotheses,
                        )

                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                        hypotheses += current_hypotheses

                    del greedy_predictions
                    del logits
                    del self_attention_outputs
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)
        return hypotheses

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
            "self_attention_outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
          Forward pass of the model.

          Args:
              input_signal: Tensor that represents a batch of raw audio signals,
                  of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                  `self.sample_rate` number of floating point values.
              input_signal_length: Vector of length B, that contains the individual lengths of the audio
                  sequences.
              processed_signal: Tensor that represents a batch of processed audio signals,
                  of shape (B, D, T) that has undergone processing via some DALI preprocessor.
              processed_signal_length: Vector of length B, that contains the individual lengths of the
                  processed audio sequences.

          Returns:
              A tuple of 4 elements -
              1) The log probabilities tensor of shape [B, T, D].
              2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
              3) The greedy token predictions of the model of shape [B, T] (via argmax)
              4) The outputs of self attention layer, of shape [B, T, D]
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len, self_attention_outputs = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        log_probs = self.decoder(encoder_output=encoded).requires_grad_()
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions, self_attention_outputs
