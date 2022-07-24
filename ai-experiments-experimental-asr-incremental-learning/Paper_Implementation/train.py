from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.collections.asr.losses.ctc import CTCLoss
import omegaconf
from omegaconf import OmegaConf
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys
import logging
from modified_model import ModifiedModel, ModifiedTeacherModel
import os
import pickle
import numpy as np

LANGUAGE = "splits"
path = '/home/DATA2/apoorvaaggarwal'
manifest_dir = os.path.join(path, LANGUAGE)
train_manifest = f"{manifest_dir}/train/train.json"
dev_manifest = f"{manifest_dir}/val/val.json"
test_manifest = f"{manifest_dir}/test/test.json"

@hydra_runner(config_path=r"/home/DATA2/apoorvaaggarwal/Paper_1_Implementation/conformer/", config_name="conformer_ctc_bpe")
def main(cfg):
    logging.debug(cfg)
    cfg['model']['train_ds']['manifest_filepath'] = train_manifest
    cfg['model']['validation_ds']['manifest_filepath'] = dev_manifest
    cfg['model']['test_ds']['manifest_filepath'] = test_manifest
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    logging.info("trainer: {}".format(cfg.trainer))
    checkpoint_callback = ModelCheckpoint(dirpath='/home/DATA2/apoorvaaggarwal/training/exp_2',
                                          save_last=True, save_top_k=20,
                                          filename='{epoch}-{val_wer:.2f}-{val_loss:.2f}',monitor="val_wer", every_n_epochs=1)
    
    checkpoint_path = None
    trainer = pl.Trainer(gpus=[0], accelerator='ddp', max_epochs=50, callbacks=[checkpoint_callback, EarlyStopping(monitor="val_wer", mode="min")], plugins=DDPPlugin(find_unused_parameters=False))
    
    teacher_model = ModifiedTeacherModel.from_pretrained("stt_en_conformer_ctc_large")
    student_model = ModifiedModel.from_pretrained("stt_en_conformer_ctc_large")

    os.chdir('/home/DATA2/apoorvaaggarwal/splits/train')
    directory = 'wav'
    manifest_dir_url = '/home/DATA2/apoorvaaggarwal/splits/train/wav'
    list_of_train_paths = []

    for filename in os.listdir(directory):
        list_of_train_paths.append(f"{manifest_dir_url}/{filename}")

#     logging.error(f"\nCURRENT DIRECTORY HAS:\n{os.listdir(directory)}")

    os.chdir('/home/DATA2/apoorvaaggarwal/splits/val')
    directory = 'wav'
    manifest_dir_url = '/home/DATA2/apoorvaaggarwal/splits/val/wav'
    list_of_dev_paths = []

    for filename in os.listdir(directory):
        list_of_dev_paths.append(f"{manifest_dir_url}/{filename}")

    list_of_all_paths = list_of_train_paths + list_of_dev_paths
    
#     logging.error(f"\nlist_of_all_paths: {list_of_all_paths}\n")
#     logging.error(f"\nCURRENT DIRECTORY HAS:\n{os.listdir(directory)}")

    os.chdir('/home/DATA2/apoorvaaggarwal/training')
    
    # Getting Teacher Model's Softmax Outputs
    teacher_logits = teacher_model.transcribe(paths2audio_files=list_of_all_paths, batch_size=2, logprobs=True)
    # Getting Teacher Model's SAB Layer Outputs as Feature Maps
    teacher_feature_map = teacher_model.transcribe(paths2audio_files=list_of_all_paths, batch_size=2,
                                                   return_self_attention_outputs=True)
    # Getting Teacher Importance Map Outputs
    teacher_importance_map = teacher_model.transcribe(paths2audio_files=list_of_all_paths, batch_size=2,
                                                   return_importance_map=True)
    

    
    # Writing objects to files to persist them
    # Writing teacher_logits object to a file
    DIR = "/home/DATA2/apoorvaaggarwal/training"
    file = f"{DIR}/teacher_logits.pkl"
    file_obj = open(file, "wb")
    # teacher_logits = list(map(lambda pred: pred.cpu().detach().numpy(), teacher_logits))
    pickle.dump({
        list_of_all_paths[idx]: teacher_logit for idx, teacher_logit in enumerate(teacher_logits)
    }, file_obj)
    file_obj.close()

    
    # Writing teacher_feature_map object to a file
    file = f"{DIR}/teacher_feature_map.pkl"
    file_obj = open(file, "wb")  # write binary
    # teacher_feature_map = list(map(lambda pred: pred.cpu().detach().numpy(), teacher_feature_map))
    pickle.dump({
        list_of_all_paths[idx]: teacher_feature for idx, teacher_feature in enumerate(teacher_feature_map)
    }, file_obj)
    file_obj.close()


    # Writing teacher_importance_map object to a file
    file = f"{DIR}/teacher_importance_map.pkl"
    file_obj = open(file, "wb")  # write binary
    # teacher_importance_map = list(map(lambda pred: pred.cpu().detach().numpy(), teacher_importance_map))
    for idx, importance_map in enumerate(teacher_importance_map):
        print(f"index when dumping: {idx} file path: {list_of_all_paths[idx]}")
                                         
    pickle.dump({
        list_of_all_paths[idx]: importance_map for idx, importance_map in enumerate(teacher_importance_map)
    }, file_obj)
    file_obj.close()

    del teacher_model
    
    # Random sample should be printed in the output at each step, along with its predicted transcript.
    student_model._wer.log_prediction = True
    
    # Setting the trainer
    student_model.set_trainer(trainer)
    
    param_config = DictConfig(cfg['model'])
    student_model.setup_training_data(param_config.train_ds)
    student_model.setup_multiple_validation_data(val_data_config=param_config.validation_ds)
    student_model.setup_multiple_test_data(test_data_config=param_config.test_ds)
    student_model.spec_augmentation = student_model.from_config_dict(student_model.cfg.spec_augment)
    student_model.setup_optimization(DictConfig(cfg['model']['optim']))
    student_model.encoder.unfreeze()
    student_model.decoder.unfreeze()
    
    trainer.fit(student_model, ckpt_path=checkpoint_path)
    checkpoint_callback.best_model_path
    checkpoint_callback.best_model_score
    trainer.save_checkpoint
    
    student_model.save_to("/home/DATA2/apoorvaaggarwal/student_model.nemo")
    
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