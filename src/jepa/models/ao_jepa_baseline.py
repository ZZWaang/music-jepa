import torch
from torch import nn
import math
from omegaconf import DictConfig
from ..components.encoders import SpVitEncoder
from ..components.baseline_jepa import JepaEmaBaselineBase
from ..components.predictors import MaskedPredictor
from ..utils.mask import get_masks
import lightning as L


class SpJepaEmaBaseline(JepaEmaBaselineBase):

    def __init__(
        self,
        sp_in_chans=1, sp_img_size=(400, 229), sp_patch_size=(25, 15), sp_embed_dim=256, sp_enc_depth=12,
        sp_enc_num_heads=4, sp_enc_mlp_ratio=2., sp_norm_layer=nn.LayerNorm, sp_spec_min_val=-80.,
        sp_pred_depth=8, sp_pred_num_heads=4, sp_pred_mlp_ratio=2.,
        student_final_norm=True, teacher_final_norm=True,
        ema_tau_base: float = 0.995, ema_tau_final: float = 0.995, ema_schedule_steps: int | None = 50_000,
        loss_type="mse", require_normalize_input=False
        ):
        super().__init__(student_final_norm=student_final_norm, teacher_final_norm=teacher_final_norm,
                         ema_tau_base=ema_tau_base, ema_tau_final=ema_tau_final, ema_schedule_steps=ema_schedule_steps,
                         loss_type=loss_type, require_normalize_input=require_normalize_input)
        encode_two = False
        self.state_student = SpVitEncoder(
            in_chans=sp_in_chans, img_size=sp_img_size, patch_size=sp_patch_size, embed_dim=sp_embed_dim,
            depth=sp_enc_depth, num_heads=sp_enc_num_heads, mlp_ratio=sp_enc_mlp_ratio, norm_layer=sp_norm_layer,
            spec_min_val=sp_spec_min_val, encode_two=encode_two
        )

        self.state_teacher = SpVitEncoder(
            in_chans=sp_in_chans, img_size=sp_img_size, patch_size=sp_patch_size, embed_dim=sp_embed_dim,
            depth=sp_enc_depth, num_heads=sp_enc_num_heads, mlp_ratio=sp_enc_mlp_ratio, norm_layer=sp_norm_layer,
            spec_min_val=sp_spec_min_val, encode_two=encode_two
        )

        self.ema_init()

        self.predictor = MaskedPredictor(
            img_size=self.state_student.padded_img_size, patch_size=sp_patch_size, embed_dim=sp_embed_dim,
            num_heads=sp_pred_num_heads, mlp_ratio=sp_pred_mlp_ratio, depth=sp_pred_depth, norm_layer=sp_norm_layer
        )


class SpJepaEmaBaselineLit(L.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.model = SpJepaEmaBaseline(
            sp_in_chans=cfg.model.sp_in_chans, sp_img_size=cfg.model.sp_img_size, sp_patch_size=cfg.model.sp_patch_size,
            sp_embed_dim=cfg.model.sp_embed_dim, sp_enc_depth=cfg.model.sp_enc_depth,
            sp_enc_num_heads=cfg.model.sp_enc_num_heads, sp_enc_mlp_ratio=cfg.model.sp_enc_mlp_ratio,
            sp_norm_layer=nn.LayerNorm, sp_spec_min_val=cfg.model.sp_spec_min_val,

            sp_pred_depth=cfg.model.sp_pred_depth, sp_pred_num_heads=cfg.model.sp_pred_num_heads,
            sp_pred_mlp_ratio=cfg.model.sp_pred_mlp_ratio,

            student_final_norm=cfg.model.student_final_norm, teacher_final_norm=cfg.model.teacher_final_norm,
            ema_tau_base=cfg.model.ema_tau_base, ema_tau_final=cfg.model.ema_tau_final, ema_schedule_steps=cfg.model.ema_schedule_steps,
            loss_type="mse",
        )
        self.log_param_stats({
            "Sp State Student": self.model.state_student,
            "Sp State Teacher": self.model.state_teacher,
            "Sp Predictor": self.model.predictor,
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

    def _split_batch(self, batch: torch.Tensor):
        spec, pr, pedal = batch
        spec = spec.unsqueeze(1)

        num_patches_sp = self.model.state_student.num_patches
        n_patches = num_patches_sp[0] * num_patches_sp[1]

        ratio = self.cfg.loader.mask_ratio
        _, sp_context_mask, sp_prediction_mask = get_masks(spec, n_patches, ratio)
        sp_context_mask = sp_context_mask.long().to(spec.device)
        sp_prediction_mask = sp_prediction_mask.long().to(spec.device)
        return spec, pr, pedal, sp_context_mask, sp_prediction_mask

    def training_step(self, batch):
        spec, _, _, sp_context_mask, sp_prediction_mask = self._split_batch(batch)

        loss, logs = self.model(
            state=spec, state_pedal=None,
            state_context_mask=sp_context_mask, state_prediction_mask=sp_prediction_mask,
            )

        # logging
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in logs.items():
            try:
                self.log(f"{k}", v, on_step=True, on_epoch=True, prog_bar=False)
            except Exception:
                pass

        return loss

    def validation_step(self, batch):
        spec, _, _, sp_context_mask, sp_prediction_mask = self._split_batch(batch)

        loss, logs = self.model.eval_batch(
            state=spec, state_pedal=None,
            state_context_mask=sp_context_mask, state_prediction_mask=sp_prediction_mask
            )

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in logs.items():
            try:
                self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=False)
            except Exception:
                pass
        return {"val_loss": loss}

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.model.update_teacher()

    def configure_optimizers(self):
        modules = [
            self.model.state_student,
            self.model.predictor,
        ]
        params = [{"params": m.parameters()} for m in modules if m is not None]

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
