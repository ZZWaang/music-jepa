import os
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from .dataset_jepa import JepaDataset


def build_jepa_dataloaders(cfg: DictConfig, data_dir="data/maestro/splits"):
    num_examples = 2 if cfg.debug else None

    dataset_cls = JepaDataset

    dataset_kwargs = dict(
        data_dir=cfg.data.data_dir,
        pr_and_pedal_h5_path=cfg.data.pr_and_pedal_h5_path,
        datasets=cfg.data.datasets,
        num_step=cfg.data.num_step,
        num_frame_per_step=cfg.data.num_frame_per_step,
        num_examples=num_examples,
        randomly_sample=False,
        num_samples_per_clip=None,
        feature_dim=229,  # only mel bins, no MFCCs
        pianoroll_dim=88,
        normalize_features=False,
        midi_chans=cfg.data.midi_chans if hasattr(cfg.data, "midi_chans") else 1,
        spec_pad_val=cfg.data.spec_pad_val if hasattr(cfg.data, "spec_pad_val") else -80.0
    )

    train_set = dataset_cls(
        data_list_path=os.path.join(data_dir, "train.json"),
        overlap_ratio=cfg.data.overlap_ratio,
        split="train",
        **dataset_kwargs
    )

    valid_set = dataset_cls(
        data_list_path=os.path.join(data_dir, "val.json"),
        overlap_ratio=0.0,
        split="validation",
        **dataset_kwargs
    )

    tl = DataLoader(train_set, batch_size=cfg.loader.batch_size, shuffle=True, num_workers=cfg.loader.num_workers, drop_last=cfg.loader.drop_last_train,
                    persistent_workers=cfg.loader.persistent_workers, pin_memory=cfg.loader.pin_memory)
    # vl shuffle is set to True for sensitivity analysis.
    vl = DataLoader(valid_set, batch_size=cfg.loader.batch_size, shuffle=True, num_workers=cfg.loader.num_workers,
                    drop_last=False, persistent_workers=cfg.loader.persistent_workers, pin_memory=cfg.loader.pin_memory)
    return tl, vl
