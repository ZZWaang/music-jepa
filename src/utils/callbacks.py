from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig


def build_jepa_callbacks(cfg: DictConfig):
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.4f}-m",
        monitor="val_loss", mode="min", save_top_k=2, save_last=True
    )
    ckpt_periodic = ModelCheckpoint(
        every_n_epochs=cfg.train.get("every_n_epochs", 5),
        save_top_k=-1,
        monitor=None,
        filename="{epoch:03d}-{val_loss:.4f}-p",
    )
    callbacks = [checkpoint_callback, ckpt_periodic]
    return callbacks
