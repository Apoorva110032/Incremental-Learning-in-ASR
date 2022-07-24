import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
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
path = '/content/drive/Othercomputers/My MacBook Pro/Google_Drive'
manifest_dir = os.path.join(path, LANGUAGE)
train_manifest = f"{manifest_dir}/train/train.json"
dev_manifest = f"{manifest_dir}/dev/dev.json"
test_manifest = f"{manifest_dir}/test/test.json"


@hydra_runner(config_path=r"/content/drive/MyDrive/Colab Notebooks/Sprinklr_Project1/conformer", config_name="conformer_ctc_bpe")
def main(cfg):
    logging.debug(cfg)
    cfg['model']['train_ds']['manifest_filepath'] = train_manifest
    cfg['model']['validation_ds']['manifest_filepath'] = dev_manifest
    cfg['model']['test_ds']['manifest_filepath'] = test_manifest

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    logging.info("trainer: {}".format(cfg.trainer))
    checkpoint_callback = ModelCheckpoint(dirpath='model_output_dir', save_last=True, save_top_k=20,
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

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None and False:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=cfg.trainer.get("amp_backend", "native"),
        )
        if asr_model.prepare_test(test_trainer):
            test_trainer.test(asr_model)


if __name__ == '__main__':
    main()
