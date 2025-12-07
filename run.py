import argparse
import yaml
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.checkpointing import get_checkpoint_path
from utils.model_factory import instantiate


# NumPy 2.0 compatibility: restore removed aliases
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_


def run(cfg_yaml):
    
    # If config has dotted keys, create from dotlist to get nested structure
    if any('.' in str(k) for k in cfg_yaml.keys()):
        # Config has flattened keys like 'logging.path'
        # We need to restructure it into nested dicts
        nested = {}
        for key, value in cfg_yaml.items():
            parts = key.split('.')
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        cfg = OmegaConf.create(nested)
    else:
        # Config already has proper nested structure
        cfg = OmegaConf.create(cfg_yaml)
    
    print(f"Config type: {type(cfg)}")
    print(f"Config keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'No keys method'}")
    print(OmegaConf.to_yaml(cfg))
    
    # Handle logging configuration - use OmegaConf.select for dotted paths
    logging_path = OmegaConf.select(cfg, 'logging.path', default='lightning_logs')
    logging_name = OmegaConf.select(cfg, 'logging.name', default='default')
    
    logger = TensorBoardLogger(save_dir=logging_path, version=logging_name, name="")    
    
    # Instantiate dataset
    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    
    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=OmegaConf.select(cfg, 'checkpoint.save_top_k', default=2),
        monitor="epoch",
        mode="max",
        filename="model-{epoch}"
    )
    
    # Set random seed - handle list of seeds or tuples
    random_seed = OmegaConf.select(cfg, 'random.seed', default=0)
    if isinstance(random_seed, (list, tuple, ListConfig)):
        # If it's a ListConfig or list, take first element
        random_seed = random_seed[0] if len(random_seed) > 0 else 0
    # Ensure it's an integer
    random_seed = int(random_seed)
    pl.seed_everything(random_seed, workers=True)
    
    # Create trainer - convert trainer config to dict and override logger
    # Make sure to resolve=True to handle any interpolations, and None values stay as None
    trainer_config = OmegaConf.select(cfg, 'trainer', default={})
    if trainer_config:
        trainer_kwargs = OmegaConf.to_container(trainer_config, resolve=True)
    else:
        trainer_kwargs = {}
    
    trainer_kwargs['logger'] = logger
    trainer_kwargs['callbacks'] = [lr_monitor, checkpoint_callback]
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Instantiate model
    model = instantiate(OmegaConf.select(cfg, 'model'), cfg=cfg)
    
    # Train
    trainer.fit(model, datamodule=dm, ckpt_path=get_checkpoint_path(cfg))
    
    # Test
    trainer.test(datamodule=dm, ckpt_path=OmegaConf.select(cfg, 'checkpoint.ckpt_path'))


def main(config_path: str, overrides: list = []):
    
    if not torch.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CUDA is NOT available !!!")
        print("!!! CUDA is NOT available !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
    with open(config_path) as f:
        cfg_yaml = yaml.unsafe_load(f)
    
    # OmegaConf can handle None values directly, no conversion needed
    # Just pass the config as-is
    run(cfg_yaml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False, description="Experiment")
    parser.add_argument('--config', type=str, 
                        help='Path to the experiment configuration file', 
                        default='config/config.yaml')
    parser.add_argument("overrides", nargs="*",
                        help="Any key=value arguments to override config values (use dots for.nested=overrides)")
    args = parser.parse_args()

    main(config_path=args.config, overrides=args.overrides)