"""Config loading and the shared command-line interface for scripts."""

import argparse
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path=None, **overrides):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--data_root", default=None, help="Folder containing the MVTec category folders")
    p.add_argument("--work_root", default=None, help="Where caches, models and results are written")
    p.add_argument("--categories", nargs="+", default=None, help="Subset of categories to run")
    p.add_argument("--force", action="store_true", help="Recompute even if cached outputs exist")
    return p


def config_from_args(args):
    cfg = load_config(args.config, data_root=args.data_root, work_root=args.work_root,
                      categories=args.categories)
    if os.environ.get("DEFECTLENS_OFFLINE") == "1":  # smoke tests without weight downloads
        cfg["patchcore"]["pretrained"] = False
    return cfg


def work_dir(cfg, *parts):
    path = Path(cfg["work_root"]).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
