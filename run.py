import argparse
import os
import yaml
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.checkpointing import get_checkpoint_path
from utils.model_factory import instantiate
from clean_data import analyze_data_quality


# NumPy 2.0 compatibility: restore removed aliases
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_


def run(cfg_yaml):
    # If config has dotted keys, create from dotlist to get nested structure
    if any('.' in str(k) for k in cfg_yaml.keys()):
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
        cfg = OmegaConf.create(cfg_yaml)

    print(f"Config type: {type(cfg)}")
    print(f"Config keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'No keys method'}")
    print(OmegaConf.to_yaml(cfg))

    logging_path = OmegaConf.select(cfg, 'logging.path', default='lightning_logs')
    base_logging_name = OmegaConf.select(cfg, 'logging.name', default='default')

    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    dm.setup("fit")
    try:
        dm.setup("test")
    except Exception as e:
        print(f"Test setup skipped: {e}")

    print("=== TRAIN DATA QUALITY ===")
    analyze_data_quality(dm.train_dataloader())
    try:
        print("=== TEST DATA QUALITY ===")
        analyze_data_quality(dm.test_dataloader())
    except Exception as e:
        print(f"Test data quality check skipped: {e}")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    seeds = OmegaConf.select(cfg, 'random.seed', default=0)
    if isinstance(seeds, (int, float)):
        seeds = [int(seeds)]
    elif isinstance(seeds, (list, tuple, ListConfig)):
        seeds = [int(s) for s in seeds]
    else:
        seeds = [0]

    for seed in seeds:
        pl.seed_everything(seed, workers=True)

        logging_name = f"{base_logging_name}-seed={seed}" if len(seeds) > 1 else base_logging_name
        logger = TensorBoardLogger(save_dir=logging_path, version=logging_name, name="")

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(logging_path, logging_name, "checkpoints"),
            save_top_k=OmegaConf.select(cfg, 'checkpoint.save_top_k', default=2),
            monitor="epoch",
            mode="max",
            filename="model-{epoch}"
        )

        trainer_config = OmegaConf.select(cfg, 'trainer', default={})
        trainer_kwargs = OmegaConf.to_container(trainer_config, resolve=True) if trainer_config else {}
        trainer_kwargs['logger'] = logger
        trainer_kwargs['callbacks'] = [lr_monitor, checkpoint_callback]

        trainer = pl.Trainer(**trainer_kwargs)

        model = instantiate(OmegaConf.select(cfg, 'model'), cfg=cfg)

        trainer.fit(model, datamodule=dm, ckpt_path=get_checkpoint_path(cfg))

        ckpt_path = checkpoint_callback.best_model_path or 'last'
        trainer.test(datamodule=dm, ckpt_path=ckpt_path)

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