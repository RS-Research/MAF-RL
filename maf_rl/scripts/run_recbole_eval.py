# maf_rl/scripts/run_recbole_eval.py
# -*- coding: utf-8 -*-
"""
MAF-RL Evaluation Entry Point (RecBole)

Usage (examples)
----------------
python -m maf_rl.scripts.run_recbole_eval --config configs/movielens1m.yaml --checkpoint saved/MAFRLModel.pth

Notes
-----
- Loads a trained model checkpoint and evaluates on the test set using RecBole metrics.
- For consistent evaluation, use the same YAML config as training.
"""

from __future__ import annotations

import argparse
import os

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed

from maf_rl.recbole_ext.models.maf_rl_model import MAFRLModel
from maf_rl.recbole_ext.trainer.ppo_trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAF-RL (RecBole) Evaluation Runner")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/movielens1m.yaml)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/cuda:0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    config = Config(
        model=MAFRLModel,
        dataset=None,
        config_file_list=[args.config],
    )
    if args.device is not None:
        config["device"] = args.device

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = config["logger"]
    logger.info(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MAFRLModel(config, train_data.dataset).to(config["device"])
    trainer = PPOTrainer(config, model)

    # Load checkpoint weights
    trainer.resume_checkpoint(args.checkpoint)

    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,   # we explicitly loaded the checkpoint
        show_progress=config.get("show_progress", True),
    )

    logger.info(f"Evaluation checkpoint: {args.checkpoint}")
    logger.info(f"Test result: {test_result}")


if __name__ == "__main__":
    main()
