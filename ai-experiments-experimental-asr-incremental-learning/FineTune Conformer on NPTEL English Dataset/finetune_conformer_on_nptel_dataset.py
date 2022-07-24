# -*- coding: utf-8 -*-
"""Finetune_Conformer_on_NPTEL_Dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m67Ii3d86Oa7iPOrqLfgqAC2hj6AjdQj

Installing Packages
"""

!pip install nemo_toolkit['all']
!pip install hydra-core==1.1
!pip install evaluate
!pip install jiwer

"""Importing Libraries"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
import omegaconf
from omegaconf import OmegaConf
from omegaconf import DictConfig
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import jiwer
import evaluate
from evaluate import load
import sys
import logging

sys.argv = ['']
del sys

"""Defining Dataset Paths"""

LANGUAGE = "slices"
path = '/content/drive/Othercomputers/My MacBook Pro/Google_Drive/Finetuning Conformer Dataset'
manifest_dir = os.path.join(path, LANGUAGE)
train_manifest = f"{manifest_dir}/train/train.json"
dev_manifest = f"{manifest_dir}/val/val.json"
test_manifest = f"{manifest_dir}/test/test.json"

"""Finetuning Conformer"""

@hydra_runner(config_path=r"/content/drive/MyDrive/Colab Notebooks/Sprinklr_Project1/conformer", config_name="conformer_ctc_bpe")
def main(cfg):
    # logging.debug(cfg)
    cfg['model']['train_ds']['manifest_filepath'] = train_manifest
    cfg['model']['validation_ds']['manifest_filepath'] = dev_manifest
    cfg['model']['test_ds']['manifest_filepath'] = test_manifest
    # logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    # logging.info("trainer: {}".format(cfg.trainer))
    checkpoint_callback = ModelCheckpoint(dirpath='/content/drive/Othercomputers/My MacBook Pro/Google_Drive/Finetuning Conformer Dataset', 
                                        save_last=True, save_top_k=20,
                                        filename='{epoch}-{val_wer:.2f}-{other_metric:.2f}',
                                        monitor="val_wer", every_n_epochs=10)
    checkpoint_path = None
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback], auto_select_gpus=True)
    asr_model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_large")
    asr_model._wer.log_prediction = True
    asr_model.set_trainer(trainer)
    param_config = DictConfig(cfg['model'])
    asr_model.setup_training_data(param_config.train_ds)
    asr_model.setup_multiple_validation_data(val_data_config=param_config.validation_ds)
    asr_model.setup_multiple_test_data(test_data_config=param_config.test_ds)
    asr_model.spec_augmentation = asr_model.from_config_dict(asr_model.cfg.spec_augment)
    asr_model.setup_optimization(DictConfig(cfg['model']['optim']))
    asr_model.encoder.unfreeze()
    asr_model.decoder.unfreeze()

    trainer.fit(asr_model, ckpt_path=checkpoint_path)
    checkpoint_callback.best_model_path
    checkpoint_callback.best_model_score
    trainer.save_checkpoint

    asr_model.save_to("/content/drive/Othercomputers/My MacBook Pro/Google_Drive/Finetuning Conformer Dataset/finetuned_model.nemo")
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None and False:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=cfg.trainer.get("amp_backend", "native"),
        )
        if asr_model.prepare_test(test_trainer):
            test_trainer.test(asr_model, verbose=True)
if __name__ == '__main__':
    main()

"""Loading Finetuned Model"""

finetuned_model = EncDecCTCModelBPE.restore_from("/content/drive/Othercomputers/My MacBook Pro/Google_Drive/Finetuning Conformer Dataset/finetuned_model.nemo")

"""Creating List of Paths"""

os.chdir('/content/drive/Othercomputers/My MacBook Pro/Google_Drive/Finetuning Conformer Dataset/slices/test')
directory = 'wav'
manifest_dir_url = '/content/drive/Othercomputers/My MacBook Pro/Google_Drive/Finetuning Conformer Dataset/slices/test/wav'
new_list_of_paths = []

for filename in os.listdir(directory):
    new_list_of_paths.append(f"{manifest_dir_url}/{filename}")
print(new_list_of_paths)

"""Creating List of actual transcripts"""

os.chdir('/content/drive/Othercomputers/My MacBook Pro/Google_Drive/Finetuning Conformer Dataset/slices/test')
directory = 'wav'
dir2 = '/content/drive/Othercomputers/My MacBook Pro/Google_Drive/Finetuning Conformer Dataset/corrected_txt'
list_of_actual_statements = []

for filename in os.listdir(directory):
    name_of_file = os.path.splitext(filename)[0]
    transcript_file_path = f"{dir2}/{name_of_file}.txt"
    file = open(transcript_file_path, 'r')

    list_of_actual_statements.append(file.read())

    file.close()

wer = load("wer")

"""Getting Predictions using Finetuned Model"""

finetuned_model_predictions = finetuned_model.transcribe(paths2audio_files=new_list_of_paths)
print(finetuned_model_predictions)

"""Initialising Pretrained Model"""

pretrained_model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_large")

"""Getting Predictions using Pretrained Model"""

pretrained_model_predictions = pretrained_model.transcribe(paths2audio_files=new_list_of_paths)

"""Removing WhiteSpace Characters"""

list_of_actual_statements = jiwer.RemoveEmptyStrings()(list_of_actual_statements)
pretrained_model_predictions = jiwer.RemoveEmptyStrings()(pretrained_model_predictions)
finetuned_model_predictions = jiwer.RemoveEmptyStrings()(finetuned_model_predictions)

"""WER using FineTuned Model on NPTEL Indian English Speech Dataset"""

wer_score = jiwer.wer(list_of_actual_statements, finetuned_model_predictions)
print(wer_score)

"""WER using Pre-trained model on NPTEL Indian English Speech Dataset"""

wer_score = jiwer.wer(list_of_actual_statements, pretrained_model_predictions)
print(wer_score)

"""Downloading Commonvoice English Dataset"""

os.chdir("/content/drive/Othercomputers/My MacBook Pro/Google_Drive/CommonVoice English Dataset")

!wget https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz/

