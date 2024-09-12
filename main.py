from dataset.dataloader_mixamo import MixamoDataModule
from dataset.dataloader_cmu import CMUDataModule
from dataset.dataloader_huamandog import HumanDogDataModule
# from dataset.dataloader_extra import MixedDataModule
from model import DATRetarget
# from dataset.dataloader_mixamo_npy import MixamoDataModule
# from model_extra import DATRetarget
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import wandb
from absl import app
from absl import flags
import warnings

from data_flags import FLAGS

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

def init_all():
    warnings.filterwarnings("ignore")

    # enable cudnn and its inbuilt auto-tuner to find the best algorithm to use for your hardware
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # useful for run-time
    torch.backends.cudnn.deterministic = True

    # pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()
    

def main(argv):
    init_all()
    project_name = "DAT HumanDog Dataset different superskel"
    # project_name = "ABL_perturb_mask"
    wandb.init(project="DATRetarget", name=project_name)
    wandb_logger = WandbLogger()
    if FLAGS.mode == "train":
        if FLAGS.dataset == 'MIXAMO':
            dm = MixamoDataModule(FLAGS)
        elif FLAGS.dataset == 'CMU':
            dm = CMUDataModule(FLAGS)
        else:
            dm = HumanDogDataModule(FLAGS)

        # dm = MixedDataModule(FLAGS)
        model = DATRetarget(FLAGS, dm)
        directory = "/home/giuliamartinelli/Code/moma/DATRetarget/checkpoint"
        if not os.path.exists(directory):
            os.makedirs(directory)

        val_checkpoint = ModelCheckpoint(monitor="Validation/loss", mode="min")
        
        trainer = Trainer(
            default_root_dir=directory,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=FLAGS.n_epochs,
            precision="bf16",
            # gradient_clip_val=5,
            # track_grad_norm=2,
            # detect_anomaly=True,
            num_sanity_val_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=1)],logger=wandb_logger)
        trainer.fit(model, dm)
    
    elif FLAGS.mode == "demo":
        dm = MixamoDataModule(FLAGS)
        model = DATRetarget(FLAGS, dm.val_dataset)
        val_checkpoint = ModelCheckpoint(monitor="Validation/loss", mode="min")
        trainer = Trainer(
            # gpus=1,
            # accelerator="auto",
            # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=FLAGS.n_epochs,
            precision="bf16",
            gradient_clip_val=5,
            track_grad_norm=2,
            # detect_anomaly=True,
            callbacks=[val_checkpoint, TQDMProgressBar(refresh_rate=1)],logger=wandb_logger)
        trainer.test(model = model,dataloaders = dm,ckpt_path=FLAGS.load_checkpoint)

if __name__ == '__main__':
    app.run(main)