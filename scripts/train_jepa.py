import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from src.utils.config import load_config
from src.dataset.dataloaders_jepa import build_jepa_dataloaders
from src.jepa.models.factory import build_lit_model
from src.utils.callbacks import build_jepa_callbacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    help="One or more YAMLs. Later ones override earlier ones.")
    ap.add_argument("--set", action="append", default=None,
                    help="Dotlist overrides, e.g. loader.batch_size=256")
    ap.add_argument("--ckpt_path", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(*args.config, set=args.set)
    
    # Optional: `--set train.seed=10`
    seed = cfg.train.get("seed")
    if seed is not None:
        L.seed_everything(seed, workers=True)

    tl, vl = build_jepa_dataloaders(cfg, data_dir=cfg.data.split_dir)

    lit = build_lit_model(cfg)

    callbacks = build_jepa_callbacks(cfg)

    csv = CSVLogger(save_dir=cfg.train.default_root_dir, name="lightning_logs")
    trainer = L.Trainer(
        logger=csv,
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.get("precision", "32-true"),
        gradient_clip_val=cfg.train.get("gradient_clip_val", 0.0),
        log_every_n_steps=cfg.train.get("log_every_n_steps", 50),
        val_check_interval=cfg.train.get("val_check_interval", None),
        default_root_dir=cfg.train.get("default_root_dir", "./exp"),
        callbacks=callbacks,
    )

    trainer.fit(lit, train_dataloaders=tl, val_dataloaders=vl, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
