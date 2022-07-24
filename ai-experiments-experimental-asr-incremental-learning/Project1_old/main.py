import os

import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.collections.asr.losses.ctc import CTCLoss
# from ctc_tried import CTCLoss
from ebkd import EBKDLoss
from rbkd import RBKDLoss
import omegaconf
from omegaconf import OmegaConf
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import sys
import logging

sys.argv = ['']
del sys

LANGUAGE = "slices"
path = '/Users/apoorvaaggarwal/Downloads/Google_Drive'
manifest_dir = os.path.join(path, LANGUAGE)
train_manifest = f"{manifest_dir}/train/train.json"
dev_manifest = f"{manifest_dir}/dev/dev.json"
test_manifest = f"{manifest_dir}/test/test.json"


class TotalLoss(CTCLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(
            num_classes=kwargs.get("num_classes"),
            zero_infinity=True,
            reduction=kwargs.get("reduction", "mean_batch")
        )
        self.CTCLoss = CTCLoss(
            num_classes=kwargs.get("num_classes"),
            zero_infinity=True,
            reduction=kwargs.get("reduction", "mean_batch"),
        )
        self.gamma = kwargs.get("gamma", 500)
        self.beta = kwargs.get("beta", 0.03)
        self.temperature = kwargs.get("temperature", 3)

        self.RBKDLoss = RBKDLoss(
            num_classes=kwargs.get("num_classes"),
            zero_infinity=True,
            reduction=kwargs.get("reduction", "mean_batch"),
            temperature=self.temperature
        )
        self.EBKDLoss = EBKDLoss(
            num_classes=kwargs.get("num_classes"),
            zero_infinity=True,
            reduction=kwargs.get("reduction", "mean_batch"),
        )

    def forward(self, log_probs, targets, input_lengths, target_lengths,
                teacher_logits, teacher_feature_map, student_feature_map):
        ctc_loss = self.CTCLoss(
            log_probs, targets, input_lengths, target_lengths
        )
        rbkd_loss = self.RBKDLoss(
            teacher_logits, log_probs
        )
        ebkd_loss = self.EBKDLoss(
            teacher_logits, log_probs, teacher_feature_map, student_feature_map
        )
        return ctc_loss + (self.beta * rbkd_loss) + (self.gamma * ebkd_loss)


@hydra_runner(config_path=r"/Users/apoorvaaggarwal/PycharmProjects/pythonProject1/conformer",
              config_name="conformer_ctc_bpe")
def main(cfg):
    # logging.debug(cfg)
    cfg['model']['train_ds']['manifest_filepath'] = train_manifest
    cfg['model']['validation_ds']['manifest_filepath'] = dev_manifest
    cfg['model']['test_ds']['manifest_filepath'] = test_manifest
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    logging.info("trainer: {}".format(cfg.trainer))
    checkpoint_callback = ModelCheckpoint(dirpath='model_output_dir', save_last=True, save_top_k=20,
                                          monitor="val_wer", every_n_epochs=10)
    checkpoint_path = None
    trainer = pl.Trainer(num_processes=1, max_epochs=100, callbacks=[checkpoint_callback])

    # Object Creation
    teacher_model = EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_large")
    # student model might have fewer params than teacher model
    student_model = EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_large")
    # batch_size
    batch_size = 32

    student_model.loss = TotalLoss(
        num_classes=student_model.decoder.num_classes_with_blank - 1,
        reduction=student_model._cfg.get("ctc_reduction", "mean_batch"),
        # hyper parameters
        temperature=3,
        gamma=500,
        beta=0.03,
    )

    student_model._wer.log_prediction = True
    student_model.set_trainer(trainer)
    param_config = DictConfig(cfg['model'])
    student_model.setup_training_data(param_config.train_ds)
    student_model.setup_multiple_validation_data(val_data_config=param_config.validation_ds)
    student_model.setup_multiple_test_data(test_data_config=param_config.test_ds)
    student_model.spec_augmentation = student_model.from_config_dict(student_model.cfg.spec_augment)
    student_model.setup_optimization(DictConfig(cfg['model']['optim']))
    student_model.encoder.unfreeze()
    student_model.decoder.unfreeze()
    trainer.fit(student_model)
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None and False:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=cfg.trainer.get("amp_backend", "native"),
        )
        if student_model.prepare_test(test_trainer):
            test_trainer.test(student_model)


if __name__ == '__main__':
    main()
