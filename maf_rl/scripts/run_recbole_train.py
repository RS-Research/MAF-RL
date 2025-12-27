# maf_rl/scripts/run_recbole_train.py
# -*- coding: utf-8 -*-
"""
MAF-RL Training Entry Point (RecBole)

Usage (examples)
----------------
python -m maf_rl.scripts.run_recbole_train --config configs/movielens1m.yaml
python -m maf_rl.scripts.run_recbole_train --config configs/amazon_books.yaml --device cuda

Notes
-----
- This script uses RecBole's standard pipeline:
    Config -> create_dataset -> data_preparation -> Trainer.fit -> Trainer.evaluate
- It registers:
    - MAFRLModel (custom model)
    - PPOTrainer (custom trainer)
- Ensure your dataset directory matches `dataset` and `data_path` in the YAML.

Repository structure expected:
- maf_rl/recbole_ext/models/maf_rl_model.py
- maf_rl/recbole_ext/trainer/ppo_trainer.py
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed

# Custom model & trainer
from maf_rl.recbole_ext.models.maf_rl_model import MAFRLModel
from maf_rl.recbole_ext.trainer.ppo_trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAF-RL (RecBole) Training Runner")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/movielens1m.yaml)")
    p.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/cuda:0)")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint (optional)")
    return p.parse_args()


def _merge_external_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Build RecBole config from YAML
    config = Config(
        model=MAFRLModel,             # allow RecBole to display model name properly
        dataset=None,                 # dataset is defined in YAML
        config_file_list=[args.config],
    )
    _merge_external_overrides(config, args)

    # Reproducibility
    init_seed(config["seed"], config["reproducibility"])

    # Logger
    init_logger(config)
    logger = config["logger"]
    logger.info(config)

    # Dataset & dataloaders
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model
    model = MAFRLModel(config, train_data.dataset).to(config["device"])

    # Trainer
    trainer = PPOTrainer(config, model)

    # Resume (optional)
    if args.checkpoint is not None:
        trainer.resume_checkpoint(args.checkpoint)

    # Fit
    best_valid_score, _ = trainer.fit(
        train_data=train_data,
        valid_data=valid_data,
        verbose=True,
        saved=True,
        show_progress=config.get("show_progress", True),
    )

    # Evaluate (test)
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config.get("show_progress", True),
    )

    logger.info(f"Best valid score: {best_valid_score}")
    logger.info(f"Test result: {test_result}")


if __name__ == "__main__":
    main()
