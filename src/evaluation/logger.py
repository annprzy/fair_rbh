import os

import neptune
import pandas as pd
import yaml
from neptune.types import File
from neptune.utils import stringify_unsupported

from src.datasets.dataset import Dataset


def init_neptune(cfg: dict):
    run = neptune.init_run(project=cfg['project'], api_token=cfg['token'])
    return run


def log_results(run, results: dict | pd.DataFrame, name: str):
    if type(results) is not pd.DataFrame:
        run[name] = stringify_unsupported(results)
    else:
        run[name].upload(File.as_html(results))


def upload_files(run, path_to_files: dict, name: str):
    for name_file, path_file in path_to_files.items():
        run[f"{name}/{name_file}"].upload(path_file)


def save_config(file_path, cfg: dict):
    with open(file_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


def save_csvs(dir_path, dataset: Dataset):
    dataset.train.to_csv(os.path.join(dir_path, 'train.csv'))
    dataset.fair.to_csv(os.path.join(dir_path, 'fair.csv'))
    dataset.test.to_csv(os.path.join(dir_path, 'test.csv'))
