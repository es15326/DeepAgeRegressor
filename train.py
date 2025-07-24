import argparse
import yaml
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import base configuration
import config
import data
import nets
from loss import WeightedAARLoss, WeightedMSELoss

def main(config_path):
    # Load model-specific configuration from the YAML file
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    # --- Model Initialization ---
    # Dynamically get the model class from the 'nets' module based on the name in the config
    model_class = getattr(nets, exp_config['model']['name'])
    model = model_class(**exp_config['model'].get('params', {}))

    # --- Data Loading ---
    data_df = pd.read_csv(config.CSV_PATH)
    train_df, val_df = train_test_split(data_df, test_size=0.1)

    img_path = config.IMG_PATH
    if config.ALIGN:
        img_path += '_aligned'

    trainset = data.GTADataset(train_df, img_path, transform=data.TRAIN_TRANSFORMS)
    valset = data.GTADataset(val_df, img_path, transform=data.EVAL_TRANSFORMS)

    # Use batch size from the experiment config, falling back to the base config
    batch_size = exp_config['training'].get('batch_size', config.BATCH_SIZE)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)

    # --- Loss Function (Optional) ---
    if 'loss' in exp_config:
        loss_name = exp_config['loss']['name']
        if loss_name == 'WeightedAARLoss':
            model.loss_func = WeightedAARLoss(data.GTADataset(train_df, '.'))
        elif loss_name == 'WeightedMSELoss':
            model.loss_func = WeightedMSELoss(data.GTADataset(train_df, '.'))

    # --- Logging and Checkpoints ---
    wandb_logger = WandbLogger(project=exp_config['logging']['project_name'])

    checkpoint_callback = ModelCheckpoint(
        monitor=exp_config['checkpoint']['monitor'],
        dirpath='data/checkpoints',
        filename=exp_config['checkpoint']['filename'],
        save_top_k=3,
        mode=exp_config['checkpoint']['mode']
    )

    # --- Trainer Initialization ---
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config.DEVICES,
        logger=wandb_logger,
        log_every_n_steps=config.LOG_STEP,
        callbacks=[checkpoint_callback]
    )

    # --- Start Training ---
    trainer.fit(model, trainloader, val_dataloaders=valloader)
    print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using a base config and an experiment-specific YAML file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the model's YAML configuration file.")
    args = parser.parse_args()
    main(args.config)
