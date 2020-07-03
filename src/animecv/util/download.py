import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

def _create_directory(path):
    dirname = os.path.dirname(path)
    Path(dirname).mkdir(parents=True, exist_ok=True)

def _download(url, path):
    res = requests.get(url)
    with open(path, mode="wb") as f:
        f.write(res.content)

def download(url, name, base_dir=None):
    logger.info(f"Download {name} from {url} to {base_dir}")
    if base_dir is None:
        home_dir = os.path.expanduser("~")
        base_dir = os.path.join(home_dir, ".animecv")

    save_path = os.path.join(base_dir, name)

    if os.path.exists(save_path):
        logger.info("File already exists.")
        return save_path
    else:
        logger.info(f"Downloading file from {url} to {save_path}")
        _create_directory(save_path)
        _download(url, save_path)
        return save_path
