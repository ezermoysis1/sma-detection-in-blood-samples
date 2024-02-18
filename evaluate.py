from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from src.train.utils import aggregate_summary


@hydra.main(config_path='config', config_name='evaluate')
def main(cfg: DictConfig) -> None:
    directory = cfg.directory
    half = os.path.abspath(directory).split('outputs/')[0]
    directory = os.path.join(half, directory)
    aggregate_summary(directory)


if __name__ == '__main__':
    main()
