from omegaconf import OmegaConf


def load_config(*config_paths, **kwargs):
    cfg = OmegaConf.create()
    for p in config_paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))

    if kwargs.get("set"):
        dot_overrides = OmegaConf.from_dotlist(kwargs["set"])
        cfg = OmegaConf.merge(cfg, dot_overrides)

    OmegaConf.resolve(cfg)

    if OmegaConf.select(cfg, "model") is None:
        cfg = cfg.cfg
    return cfg
