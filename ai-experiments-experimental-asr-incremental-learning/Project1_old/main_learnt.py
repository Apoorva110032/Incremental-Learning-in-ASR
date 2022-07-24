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


@hydra_runner(config_path=r"/Users/apoorvaaggarwal/PycharmProjects/pythonProject1/conformer",
              config_name="conformer_ctc_bpe")
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
    trainer = pl.Trainer(num_processes=1, max_epochs=100, callbacks=[checkpoint_callback])

    # Object Creation
    teacher_model = EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")
    # student model might have fewer params than teacher model
    student_model = EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")
    # batch_size
    batch_size = 32
    # hyper parameters
    temperature = 3
    beta = 0.03
    gamma = 500

    # ----Should be inside training loop or training method function call----
    # # forward pass in teacher model
    # teacher_logits = teacher_model.transcribe(
    #     paths2audio_files=f"{manifest_dir}/train/recordings",
    #     batch_size=batch_size,
    #     logprobs=True)
    # # forward pass in student model
    # student_logits = student_model.transcribe(
    #     paths2audio_files=f"{manifest_dir}/train/recordings",
    #     batch_size=batch_size,
    #     logprobs=True)

    # # assuming logits shape is batch_size x sequence_length x vocab_size
    #
    # batch_size = teacher_logits.shape[0]
    # sequence_length = teacher_logits.shape[1]
    # vocab_size = teacher_logits.shape[2]

    # # Loss Computation
    # del student_model.loss
    # ctc_loss = CTCLoss(
    #     num_classes=student_model.decoder.num_classes_with_blank - 1,
    #     zero_infinity=True
    # )

    student_model._wer.log_prediction = True
    student_model.set_trainer(trainer)
    param_config = DictConfig(cfg['model'])
    student_model.setup_training_data(param_config.train_ds)
    student_model.cuda()

    # Figure out how to get the parameter values 
    # loss_ctc = torch.tensor()
    # for train_batch in student_model.temporary_datalayer():
    #     train_batch = [x.cuda() for x in train_batch]
    #     targets = train_batch[2]
    #     targets_lengths = train_batch[3]
    #     log_probs, encoded_len, greedy_predictions = student_model.forward(
    #         input_signal=train_batch[0], input_signal_length=train_batch[1]
    #     )
    # 
    #     loss_ctc = torch.cat(loss_ctc, ctc_loss(log_probs=log_probs,
    #                                             targets=targets,
    #                                             input_lengths=encoded_len,
    #                                             target_lengths=targets_lengths))
    # 
    # final_ctc_loss = torch.mean(loss_ctc)
    # student_model.loss = final_ctc_loss + \
    #                 (gamma * EBKDLoss(teacher_logits, student_logits, teacher_feature_map, student_feature_map)) + \
    #                 (beta * RBKDLoss(teacher_logits, student_logits, temperature))

    #  * train_ds - to instantiate training dataset
    #  * validation_ds - to instantiate validation dataset
    #  * test_ds - to instantiate testing dataset
    #  * optim - to instantiate optimizer with learning rate scheduler
    student_model.setup_multiple_validation_data(val_data_config=param_config.validation_ds)
    student_model.setup_multiple_test_data(test_data_config=param_config.test_ds)
    
    # wer_nums = []
    # wer_denoms = []
    # 
    # for test_batch in student_model.test_dataloader():
    #     test_batch = [x.cuda() for x in test_batch]
    #     targets = test_batch[2]
    #     targets_lengths = test_batch[3]
    #     log_probs, encoded_len, greedy_predictions = student_model(
    #         input_signal=test_batch[0], input_signal_length=test_batch[1]
    #     )
    #     # Notice the model has a helper object to compute WER
    #     student_model._wer.update(greedy_predictions, targets, targets_lengths)
    #     _, wer_num, wer_denom = student_model._wer.compute()
    #     wer_nums.append(wer_num.detach().cpu().numpy())
    #     wer_denoms.append(wer_denom.detach().cpu().numpy())
    # 
    # # We need to sum all numerators and denominators first. Then divide.
    # print(f"WER = {sum(wer_nums) / sum(wer_denoms)}")

    student_model.spec_augmentation = student_model.from_config_dict(student_model.cfg.spec_augment)
    student_model.setup_optimization(DictConfig(cfg['model']['optim']))
    student_model.encoder.unfreeze()
    student_model.decoder.unfreeze()
    trainer.fit(student_model, ckpt_path=checkpoint_path)
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


if _name_ == '_main_':
    main()
