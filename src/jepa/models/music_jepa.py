import torch
from torch import nn
import math
from omegaconf import DictConfig
from ..components.predictors import MaskedPredictorWithAction
from ..components.encoders import SpVitEncoder, PrVitEncoder
from ..components.base_jepa import JepaEmaBase
from ..components.predictors import MaskedPredictor
from ..utils.mask import get_fixed_temporal_masks
import lightning as L


class SpJepaEma(JepaEmaBase):

    def __init__(
        self,
        sp_in_chans=1, sp_img_size=(400, 229), sp_patch_size=(25, 15), sp_embed_dim=256, sp_enc_depth=12,
        sp_enc_num_heads=4, sp_enc_mlp_ratio=2., sp_norm_layer=nn.LayerNorm, sp_spec_min_val=-80.,
        pr_in_chans=2, pr_img_size=(400, 88), pr_patch_size=(25, 8), pr_embed_dim=256, pr_enc_depth=12,
        pr_enc_num_heads=4, pr_enc_mlp_ratio=2., pr_norm_layer=nn.LayerNorm,
        sp_pred_depth=8, sp_pred_num_heads=4, sp_pred_mlp_ratio=2.,
        pr_pred_depth=6, pr_pred_num_heads=4, pr_pred_mlp_ratio=2.,
        student_final_norm=True, teacher_final_norm=True,
        ema_tau_base: float = 0.95, ema_tau_final: float = 0.95, ema_schedule_steps: int | None = 50_000,
        loss_type="mse", train_action_prior=False, random_mask=True, prior_loss_weight=1.0,
        require_normalize_input=False, action_target_encoder="ema"
        ):
        super().__init__(student_final_norm=student_final_norm, teacher_final_norm=teacher_final_norm,
                         ema_tau_base=ema_tau_base, ema_tau_final=ema_tau_final, ema_schedule_steps=ema_schedule_steps,
                         loss_type=loss_type, train_action_prior=train_action_prior,
                         random_mask=random_mask, prior_loss_weight=prior_loss_weight,
                         require_normalize_input=require_normalize_input,
                         action_target_encoder=action_target_encoder)

        # This class always encodes two states at the same time.
        # E.g., for sp_img_size=(400, 229), effective state size is (200, 229).
        encode_two = True

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

        self.action_student = PrVitEncoder(
            in_chans=pr_in_chans, img_size=pr_img_size, patch_size=pr_patch_size, embed_dim=pr_embed_dim,
            depth=pr_enc_depth, num_heads=pr_enc_num_heads, mlp_ratio=pr_enc_mlp_ratio, norm_layer=pr_norm_layer,
            encode_two=encode_two
        )

        if self.train_action_prior and self.action_target_encoder == "ema":
            self.action_teacher = PrVitEncoder(
                in_chans=pr_in_chans, img_size=pr_img_size, patch_size=pr_patch_size, embed_dim=pr_embed_dim,
                depth=pr_enc_depth, num_heads=pr_enc_num_heads, mlp_ratio=pr_enc_mlp_ratio, norm_layer=pr_norm_layer,
                encode_two=encode_two
            )
        else:
            self.action_teacher = None

        self.ema_init()

        self.predictor = MaskedPredictorWithAction(
            img_size=self.state_student.padded_img_size, patch_size=sp_patch_size, embed_dim=sp_embed_dim,
            num_heads=sp_pred_num_heads, mlp_ratio=sp_pred_mlp_ratio, depth=sp_pred_depth, norm_layer=sp_norm_layer
        )

        if self.train_action_prior:
            self.prior_predictor = MaskedPredictor(
            img_size=self.action_student.padded_img_size, patch_size=pr_patch_size, embed_dim=pr_embed_dim,
            num_heads=pr_pred_num_heads, mlp_ratio=pr_pred_mlp_ratio, depth=pr_pred_depth, norm_layer=pr_norm_layer
        )
        else:
            self.prior_predictor = None


class SpJepaEmaLit(L.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.model = SpJepaEma(
            sp_in_chans=cfg.model.sp_in_chans, sp_img_size=cfg.model.sp_img_size, sp_patch_size=cfg.model.sp_patch_size,
            sp_embed_dim=cfg.model.sp_embed_dim, sp_enc_depth=cfg.model.sp_enc_depth,
            sp_enc_num_heads=cfg.model.sp_enc_num_heads, sp_enc_mlp_ratio=cfg.model.sp_enc_mlp_ratio,
            sp_norm_layer=nn.LayerNorm, sp_spec_min_val=cfg.model.sp_spec_min_val,

            pr_in_chans=cfg.model.pr_in_chans, pr_img_size=cfg.model.pr_img_size, pr_patch_size=cfg.model.pr_patch_size,
            pr_embed_dim=cfg.model.pr_embed_dim, pr_enc_depth=cfg.model.pr_enc_depth,
            pr_enc_num_heads=cfg.model.pr_enc_num_heads, pr_enc_mlp_ratio=cfg.model.pr_enc_mlp_ratio, pr_norm_layer=nn.LayerNorm,

            sp_pred_depth=cfg.model.sp_pred_depth, sp_pred_num_heads=cfg.model.sp_pred_num_heads,
            sp_pred_mlp_ratio=cfg.model.sp_pred_mlp_ratio,
            pr_pred_depth=cfg.model.pr_pred_depth, pr_pred_num_heads=cfg.model.pr_pred_num_heads,
            pr_pred_mlp_ratio=cfg.model.pr_pred_mlp_ratio,
            student_final_norm=cfg.model.student_final_norm, teacher_final_norm=cfg.model.teacher_final_norm,
            ema_tau_base=cfg.model.ema_tau_base, ema_tau_final=cfg.model.ema_tau_final, ema_schedule_steps=cfg.model.ema_schedule_steps,
            loss_type="mse",
            train_action_prior=cfg.model.train_action_prior,
            random_mask=cfg.model.random_mask,
            prior_loss_weight=cfg.model.prior_loss_weight,
            action_target_encoder=cfg.model.get("action_target_encoder", "ema"),
        )

        self.log_param_stats({
            "Sp State Student": self.model.state_student,
            "Sp State Teacher": self.model.state_teacher,
            "Sp Predictor": self.model.predictor,
            "Pr Action Student": self.model.action_student,
            "Pr Action Teacher": self.model.action_teacher if self.model.action_teacher is not None else torch.nn.Identity(),
            "Pr Prior Predictor": self.model.prior_predictor if self.model.prior_predictor is not None else torch.nn.Identity()
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
        if pr.dim() == 3:
            pr = pr.unsqueeze(1)
        pedal = pedal.unsqueeze(1)

        num_patches_sp = self.model.state_student.num_patches
        num_patches_pr = self.model.action_student.num_patches

        ratio = self.cfg.loader.mask_ratio

        if self.cfg.model.random_mask:
            sp_context_mask, sp_prediction_mask, pr_context_mask, pr_prediction_mask = \
                get_fixed_temporal_masks(
                    spec, num_patches_sp, num_patches_pr, ratio=ratio,
                    device=spec.device, mask_type="random"
                    )
        else:
            sp_context_mask, sp_prediction_mask, pr_context_mask, pr_prediction_mask = \
                get_fixed_temporal_masks(
                    spec, num_patches_sp, num_patches_pr, ratio=[0.5, 0.5],
                    device=spec.device, mask_type="temporal_prev"
                    )
        sp_context_mask = sp_context_mask.long().to(spec.device)
        sp_prediction_mask = sp_prediction_mask.long().to(spec.device)
        pr_context_mask = pr_context_mask.long().to(spec.device)
        pr_prediction_mask = pr_prediction_mask.long().to(spec.device)
        return spec, pr, pedal, sp_context_mask, sp_prediction_mask, pr_context_mask, pr_prediction_mask

    def training_step(self, batch):
        spec, pr, pedal, sp_context_mask, sp_prediction_mask, pr_context_mask, pr_prediction_mask = self._split_batch(batch)

        loss, logs = self.model(
            state=spec, action=pr, state_pedal=None, action_pedal=pedal,
            state_context_mask=sp_context_mask, state_prediction_mask=sp_prediction_mask,
            action_context_mask=pr_context_mask, action_prediction_mask=pr_prediction_mask
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
        spec, pr, pedal, sp_context_mask, sp_prediction_mask, pr_context_mask, pr_prediction_mask = self._split_batch(batch)

        loss, logs = self.model.eval_batch(
            state=spec, action=pr, state_pedal=None, action_pedal=pedal,
            state_context_mask=sp_context_mask, state_prediction_mask=sp_prediction_mask,
            action_context_mask=pr_context_mask, action_prediction_mask=pr_prediction_mask
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
            self.model.action_student,
            self.model.predictor,
            self.model.prior_predictor
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
