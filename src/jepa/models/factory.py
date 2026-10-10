from collections.abc import Callable

import lightning as L
from omegaconf import DictConfig
from ...utils.config import load_config
from .music_jepa import SpJepaEmaLit
from .ao_jepa_baseline import SpJepaEmaBaselineLit
from .state_decoder import SpectrogramVitDecoderLit
from .action_decoder import PianoRollVitDecoderTranscriptionLit
from .inverse_model import PrInversePredictorLit
from .action_dae import PrDaeLit
from .forward_model import SpPredictorLit


def build_music_jepa(cfg: DictConfig) -> L.LightningModule:
    """Build the main Music-JEPA model."""
    return SpJepaEmaLit(cfg)


def build_music_jepa_baseline(cfg: DictConfig) -> L.LightningModule:
    """Build the state-only Music-JEPA baseline."""
    return SpJepaEmaBaselineLit(cfg)


def _pretrain_path(cfg: DictConfig, key: str) -> str:
    """Return cfg.pretrain.<key>, failing clearly when it is not set."""
    path = cfg.pretrain.get(key)
    if path is None:
        raise ValueError(f"pretrain.{key} is not set; pass it with --set pretrain.{key}=<path>")
    return path


def _load_pretrained_music_jepa(cfg: DictConfig) -> SpJepaEmaLit:
    """Load the Music-JEPA checkpoint specified by cfg.pretrain."""
    pretrain_cfg = load_config(_pretrain_path(cfg, "jepa_cfg_path"))
    return SpJepaEmaLit.load_from_checkpoint(
        _pretrain_path(cfg, "jepa_ckpt_path"),
        cfg=pretrain_cfg,
    )


def build_spectrogram_decoder(cfg: DictConfig) -> L.LightningModule:
    """Build a spectrogram decoder on the pretrained state encoder."""
    jepa = _load_pretrained_music_jepa(cfg)

    return SpectrogramVitDecoderLit(
        low_level_enc=jepa.model.state_student,
        cfg=cfg,
    )


def build_piano_roll_transcription_decoder(cfg: DictConfig) -> L.LightningModule:
    """Build a piano-roll decoder (w/ piano transcription loss) on the action encoder."""
    jepa = _load_pretrained_music_jepa(cfg)

    return PianoRollVitDecoderTranscriptionLit(
        low_level_enc=jepa.model.action_student,
        cfg=cfg,
    )


def build_inverse_predictor(cfg: DictConfig) -> L.LightningModule:
    """Build an inverse predictor on frozen or fine-tunable JEPA encoders."""
    jepa = _load_pretrained_music_jepa(cfg)

    return PrInversePredictorLit(
        state_enc=jepa.model.state_student,
        action_enc=jepa.model.action_student,
        cfg=cfg,
    )


def build_piano_roll_dae(cfg: DictConfig) -> L.LightningModule:
    """Build a piano-roll denoising autoencoder with pretrained JEPA modules."""
    jepa = _load_pretrained_music_jepa(cfg)

    inverse_cfg = load_config(_pretrain_path(cfg, "inv_cfg_path"))
    inverse_lit = PrInversePredictorLit.load_from_checkpoint(
        _pretrain_path(cfg, "inv_ckpt_path"),
        cfg=inverse_cfg,
        state_enc=jepa.model.state_student,
        action_enc=jepa.model.action_student,
    )

    return PrDaeLit(
        state_enc=jepa.model.state_student,
        action_enc=jepa.model.action_student,
        inv_pred=inverse_lit.model,
        cfg=cfg,
    )


def build_state_predictor(cfg: DictConfig) -> L.LightningModule:
    """Build a state predictor on pretrained JEPA encoders."""
    jepa = _load_pretrained_music_jepa(cfg)

    return SpPredictorLit(
        state_enc=jepa.model.state_student,
        action_enc=jepa.model.action_student,
        cfg=cfg,
    )


MODEL_BUILDERS: dict[str, Callable[[DictConfig], L.LightningModule]] = {
    "jepa": build_music_jepa,
    "jepa-baseline": build_music_jepa_baseline,
    "spectrogram-decoder": build_spectrogram_decoder,
    "piano-roll-transcription-decoder": build_piano_roll_transcription_decoder,
    "inverse-predictor": build_inverse_predictor,
    "state-predictor": build_state_predictor,
    "piano-roll-dae": build_piano_roll_dae,
}


def build_lit_model(cfg: DictConfig) -> L.LightningModule:
    """Build the Lightning module selected by cfg.model.name."""
    name = cfg.model.name.lower()

    try:
        return MODEL_BUILDERS[name](cfg)
    except KeyError as error:
        available = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(
            f"Unknown model name {name!r}. Available models: {available}"
        ) from error
