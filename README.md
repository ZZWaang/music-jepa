# Music-JEPA

This is the code repository for the paper **[Music-JEPA: Learning a World Model of Piano Sound from Action](https://arxiv.org/abs/2607.22000)**.
Music-JEPA learns audio representations from spectrogram states and piano-roll actions
using a Joint Embedding Predictive Architecture (JEPA). Listen to audio examples on the
[demo page](https://zzwaang.github.io/music-jepa-demo/).

We provide training code, data, and pretrained checkpoints for Music-JEPA and its variants,
together with an audio-only JEPA baseline (AO-JEPA).

## Environment

We use Python 3.10. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data

The dataset contains preprocessed audio features, aligned MIDI piano rolls, and sustain
pedal values derived from [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro).
It covers 1,276 piano performances and approximately 200 hours of audio.
Download it to `data/maestro/`:

```bash
hf download ZzWaang/music-jepa-maestro --type dataset --local-dir data/maestro
```

This downloads `maestro_audio.h5` (audio features, about 45 GB), `maestro_midi.h5`
(piano rolls and sustain pedal values), and `splits/{train,val,test}.json`
(performance metadata for the official MAESTRO splits).
See the [dataset card](https://huggingface.co/datasets/ZzWaang/music-jepa-maestro) for file formats and metadata details.

## Checkpoints

Download the checkpoints to `checkpoints/`:

```bash
hf download ZzWaang/music-jepa --local-dir checkpoints
```

| Folder | Model |
| --- | --- |
| `music_jepa/` | Music-JEPA (main model) |
| `music_jepa_prior_ema/` | Music-JEPA + action predictor, EMA action target |
| `music_jepa_prior_student/` | Music-JEPA + action predictor, student action target |
| `ao_jepa_baseline/` | Audio-only JEPA baseline (AO-JEPA) |
| `music_jepa/bundle/*` | Inverse and forward models, state and action decoders, and an action-latent denoising autoencoder, trained on the frozen Music-JEPA |

Each folder contains `model.ckpt` and the `config.yaml` used to train it.
All models are trained on the MAESTRO training split.
See the [model card](https://huggingface.co/ZzWaang/music-jepa) for details.

To load a model:

```python
from src.jepa.models.music_jepa import SpJepaEmaLit
model = SpJepaEmaLit.load_from_checkpoint("checkpoints/music_jepa/model.ckpt")
```

## Training

To train the JEPA models, run all commands from the repository root:

```bash
# Train Music-JEPA
python scripts/train_jepa.py --config configs/jepa/music_jepa.yaml

# Train the action-predictor variant with EMA targets
python scripts/train_jepa.py --config configs/jepa/music_jepa_prior_ema.yaml

# Train the action-predictor variant with student targets
python scripts/train_jepa.py --config configs/jepa/music_jepa_prior_student.yaml

# Train the audio-only JEPA baseline
python scripts/train_jepa.py --config configs/jepa/ao_jepa_baseline.yaml
```

We train additional modules for inference and decoding on top of a frozen, pretrained
Music-JEPA model. Together, these modules form a bundle. Set the paths to the pretrained
JEPA config and checkpoint, then run the training commands below:

```bash
# Train the inverse model
python scripts/train_jepa.py --config configs/bundle/inverse_model.yaml \
  --set pretrain.jepa_cfg_path=checkpoints/music_jepa/config.yaml \
  --set pretrain.jepa_ckpt_path=checkpoints/music_jepa/model.ckpt

# Train the forward model
python scripts/train_jepa.py --config configs/bundle/forward_model.yaml \
  --set pretrain.jepa_cfg_path=checkpoints/music_jepa/config.yaml \
  --set pretrain.jepa_ckpt_path=checkpoints/music_jepa/model.ckpt

# Train the state decoder
python scripts/train_jepa.py --config configs/bundle/state_decoder.yaml \
  --set pretrain.jepa_cfg_path=checkpoints/music_jepa/config.yaml \
  --set pretrain.jepa_ckpt_path=checkpoints/music_jepa/model.ckpt

# Train the action decoder
python scripts/train_jepa.py --config configs/bundle/action_decoder.yaml \
  --set pretrain.jepa_cfg_path=checkpoints/music_jepa/config.yaml \
  --set pretrain.jepa_ckpt_path=checkpoints/music_jepa/model.ckpt

# Train the action-latent denoising autoencoder
python scripts/train_jepa.py --config configs/bundle/action_dae.yaml \
  --set pretrain.jepa_cfg_path=checkpoints/music_jepa/config.yaml \
  --set pretrain.jepa_ckpt_path=checkpoints/music_jepa/model.ckpt \
  --set pretrain.inv_cfg_path=checkpoints/music_jepa/bundle/inverse_model/config.yaml \
  --set pretrain.inv_ckpt_path=checkpoints/music_jepa/bundle/inverse_model/model.ckpt
```

Note that the action DAE also requires a trained inverse model, given by `pretrain.inv_*`.

Any config value can be overridden with `--set`. For example:

- `--set debug=true` trains on two tracks only, for a quick check.
- `--set loader.batch_size=16` changes the batch size.
- `--set loader.num_workers=0 --set loader.persistent_workers=false` loads data in a single process.

Logs and checkpoints are written to `train.default_root_dir` (under `lit_exp/` by default).

## Notes

- Our original code is released under the [MIT License](LICENSE). Third-party code
  retains its upstream license: `src/jepa/utils/tensors.py` comes from
  [V-JEPA 2](https://github.com/facebookresearch/vjepa2) (MIT);
  `src/jepa/layers/pos_embed.py` comes from
  [Stem-JEPA](https://github.com/SonyCSLParis/Stem-JEPA) (LGPL-3.0), originally from
  [MAE](https://github.com/facebookresearch/mae) (CC BY-NC 4.0); and the patch embeddings in
  `src/jepa/layers/patch_embed.py` are adapted from
  [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0).
- Code for downstream experiments will be added in a future update.

## Citation

If you use this code, please cite Music-JEPA:

```bibtex
@article{wang2026musicjepa,
  title   = {{Music-JEPA}: Learning a World Model of Sound from Action},
  author  = {Ziyu Wang and Kun Fang and Yann LeCun},
  journal = {CoRR},
  volume  = {abs/2607.22000},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.22000},
  url     = {https://arxiv.org/abs/2607.22000}
}
```
