import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import math
import lightning as L
from ..layers.transformers import VisionTransformer


class PrDae(nn.Module):

    def __init__(
        self,
        pr_padded_img_size=(400, 90), pr_patch_size=(25, 6),
        pr_norm_layer=nn.LayerNorm,
        pr_embed_dim=256, pr_pred_depth=6, pr_pred_num_heads=4, pr_pred_mlp_ratio=2.,
        noise_std=0.12, pred_endpoint_prob=None):
        super().__init__()

        self.predictor = VisionTransformer(
            img_size=pr_padded_img_size, patch_size=pr_patch_size, embed_dim=pr_embed_dim,
            num_heads=pr_pred_num_heads, mlp_ratio=pr_pred_mlp_ratio, depth=pr_pred_depth, norm_layer=pr_norm_layer,
            output_dim=pr_embed_dim
        )
        self.noise_std = noise_std
        if pred_endpoint_prob is not None and not 0.0 <= pred_endpoint_prob <= 1.0:
            raise ValueError("pred_endpoint_prob must be between 0 and 1, or None.")
        self.pred_endpoint_prob = pred_endpoint_prob

    def predict(self, a_noise):
        if a_noise.dim() == 3:  # (B, H*W, D) -> (B, H, W, D)
            a_noise = a_noise.reshape(
                a_noise.size(0),
                self.predictor.grid_size[0] // 2,
                self.predictor.grid_size[1],
                a_noise.size(-1)
            )
        return self.predictor(a_noise)

    def forward(self, a_curr, a_pred):
        # a_curr: target / clean action
        # a_pred: predicted action to be projected back

        # 1) local denoising around ground-truth action
        a_noise = a_curr + self.noise_std * torch.randn_like(a_curr)
        recon_a_noise = self.predict(a_noise)
        loss_noise = F.mse_loss(recon_a_noise, a_curr)

        # 2) denoising / projection for predicted action
        alpha_shape = [a_curr.size(0)] + [1] * (a_curr.dim() - 1)
        alpha = torch.rand(alpha_shape, device=a_curr.device, dtype=a_curr.dtype)

        if self.pred_endpoint_prob is not None:
            use_pred_endpoint = (
                torch.rand(alpha_shape, device=a_curr.device)
                < self.pred_endpoint_prob
            )
            alpha = torch.where(use_pred_endpoint, torch.zeros_like(alpha), alpha)

        a_mix = alpha * a_curr + (1.0 - alpha) * a_pred
        recon_a_pred = self.predict(a_mix)
        loss_pred = F.mse_loss(recon_a_pred, a_curr)

        loss = loss_noise + 2.0 * loss_pred

        log = {
            "loss": loss.detach(),
            "loss_noise": loss_noise.detach(),
            "loss_pred": loss_pred.detach(),
        }

        return loss, log


class PrDaeLit(L.LightningModule):

    def __init__(self, state_enc, action_enc, inv_pred, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['state_enc', "action_enc", "inv_pred"])

        self.cfg = cfg
        self.state_enc = state_enc
        self.action_enc = action_enc
        self.inv_pred = inv_pred
        self.state_enc.eval()
        self.action_enc.eval()
        self.inv_pred.eval()

        self.require_normalize_input = False
        self.require_norm = True

        for params in self.state_enc.parameters():
            params.requires_grad = False

        for params in self.action_enc.parameters():
            params.requires_grad = False

        for params in self.inv_pred.parameters():
            params.requires_grad = False

        self.model = PrDae(
            pr_padded_img_size=self.action_enc.padded_img_size,
            pr_patch_size=self.action_enc.patch_size,
            pr_embed_dim=cfg.model.embed_dim,
            pr_pred_depth=cfg.model.depth,
            pr_pred_num_heads=cfg.model.num_heads,
            pr_pred_mlp_ratio=cfg.model.mlp_ratio,
            pr_norm_layer=nn.LayerNorm,
            noise_std=cfg.model.noise_std,
            pred_endpoint_prob=cfg.model.get("pred_endpoint_prob", None),
        )

        self.log_param_stats({
            "Sp State Student": self.state_enc,
            "Pr Action Student": self.action_enc,
            "Inverse Predictor": self.inv_pred,
            "Pr DAE": self.model,
        })

    def log_param_stats(self, modules: dict):

        def count_parameters(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return {
                "total": total,
                "trainable": trainable,
                "non_trainable": total - trainable,
            }

        stats = {name: count_parameters(m) for name, m in modules.items()}

        for name, s in stats.items():
            print(f"{name}: total={s['total']}, trainable={s['trainable']}, non-trainable={s['non_trainable']}")

        total = {k: sum(s[k] for s in stats.values()) for k in ["total", "trainable", "non_trainable"]}

        print(f"Sum: total={total['total']}, trainable={total['trainable']}, non-trainable={total['non_trainable']}")
        print('-' * 94)

    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint["state_dict"]

        drop_keys = [k for k in sd.keys() if
                        k.startswith("state_enc.") or
                        k.startswith("action_enc.") or
                        k.startswith("inv_pred.")
                    ]
        for k in drop_keys:
            sd.pop(k, None)

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, **kwargs):
        return super().load_from_checkpoint(ckpt_path, strict=False, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = False):
        pruned = {k: v for k, v in state_dict.items()
                  if not k.startswith("state_enc.") and not k.startswith("action_enc.") and not k.startswith("inv_pred.")}
        return super().load_state_dict(pruned, strict=False)

    def _split_batch(self, batch: torch.Tensor):
        spec, pr, pedal = batch
        spec = spec.unsqueeze(1)
        pr = pr.unsqueeze(1)
        pedal = pedal.unsqueeze(1)
        return spec, pr, pedal

    def encode_state_action(self, spec, pr, pedal):
        # encode state and action
        s_full = self.state_enc(spec, mask=None,
                                require_normalize_input=self.require_normalize_input)
        if self.require_norm:
            s_full = F.layer_norm(s_full, (s_full.size(-1),))

        a_full = self.action_enc(pr, pedal=pedal, mask=None,
                                 require_normalize_input=self.require_normalize_input)
        if self.require_norm:
            a_full = F.layer_norm(a_full, (a_full.size(-1),))
        a_prev = a_full[:, :a_full.size(1) // 2]
        a_curr = a_full[:, a_full.size(1) // 2:]

        a_pred = self.inv_pred.predict(a_prev, s_full)
        return a_curr, a_pred

    def training_step(self, batch):
        spec, pr, pedal = self._split_batch(batch)

        with torch.no_grad():
            a_curr, a_pred = self.encode_state_action(spec, pr, pedal)

        loss, logs = self.model(a_curr, a_pred)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in logs.items():
            try:
                self.log(f"{k}", v, on_step=True, on_epoch=True, prog_bar=False)
            except Exception:
                pass

        return loss

    def validation_step(self, batch):
        spec, pr, pedal = self._split_batch(batch)

        with torch.no_grad():
            a_curr, a_pred = self.encode_state_action(spec, pr, pedal)

        loss, logs = self.model(a_curr, a_pred)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in logs.items():
            try:
                self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=False)
            except Exception:
                pass
        return {"val_loss": loss}

    def configure_optimizers(self):
        params = [
            {"params": self.model.parameters()},
        ]
        opt = torch.optim.AdamW(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

        sched_name = self.cfg.train.sched
        assert sched_name == "cosine"

        total_steps = self.trainer.estimated_stepping_batches or 1
        warmup = max(1, self.cfg.train.warmup_steps)
        base, floor = self.cfg.train.lr, self.cfg.train.min_lr

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (floor / base) + (1.0 - floor / base) * cos

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
            },
        }
