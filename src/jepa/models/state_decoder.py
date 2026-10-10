import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import math
import lightning as L
from ..layers.transformers import VisionTransformer
from ..layers.patch_embed import PatchUnembed
from ..layers.attention import AttentivePoolingBlock


class SpectrogramVitDecoder(nn.Module):

    def __init__(self, img_size=(400, 229), patch_size=(25, 15), input_dim=256, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4.,
                 deconv_dim=256, out_chans=1, norm_layer=nn.LayerNorm, spec_min_val=-80.0,
                 expand_input_with_attn=False):
        super().__init__()
        self.patch_size = patch_size

        self.img_size=img_size
        self.num_patches = (int(math.ceil(img_size[0] / patch_size[0])), int(math.ceil(img_size[1] / patch_size[1])))
        self.padded_img_size = (self.num_patches[0] * patch_size[0], self.num_patches[1] * patch_size[1])
        self.spec_min_val = spec_min_val

        self.in_linear = nn.Linear(input_dim, embed_dim)

        self.pad_t, self.pad_f = self.padded_img_size[0] - img_size[0], self.padded_img_size[1] - img_size[1]

        self.backbone = VisionTransformer(
            img_size=self.padded_img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )

        self.unembed = PatchUnembed(patch_size=patch_size, embed_dim=embed_dim,
                                    deconv_dim=deconv_dim, out_chans=out_chans, norm_layer=None)

        if expand_input_with_attn:
            self.input_attn = AttentivePoolingBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                    norm_layer=norm_layer, num_query=self.num_patches[1])
        else:
            self.input_attn = None
        self.in_linear.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        if self.input_attn is None:
            if z.dim() == 3:
                z = z.reshape(z.size(0), self.num_patches[0], self.num_patches[1], z.size(-1))  # (B, TT, PP, D)
            h = self.in_linear(z)  # (B, TT, PP, 256)
            B, Hs, Ws, D = z.shape
        else:
            assert z.dim() == 4  # (B, TT, 1, D)
            h = self.in_linear(z)  # (B, TT, 1, 256)
            B, Hs, _, D = h.shape
            h = self.input_attn(h.reshape(B * Hs, 1, D))  # B * Hs, Ws, D
            h = h.reshape(B, Hs, -1, D)
            B, Hs, Ws, D = h.shape

        h = self.backbone(h)  # (B, TT * PP, 256)
        h = h.view(B, Hs, Ws, -1)  # (B, TT, PP, 256)
        x = self.unembed(h)  # (B, 1, Padded_T, Padded_P)
        x = torch.tanh(x)
        x = x[:, :, :self.img_size[0], :self.img_size[1]].contiguous()  # crop to original size (B, 1, T, P)
        return x

    def loss_function(self, recon_x, x, eps=1e-8, alpha=0.5):
        """
        normalize x to [-1, 1] by x' = -2 * x / spec_min_val + 1
        """
        assert recon_x.dim() == 4 and x.dim() == 4

        x = 2.0 * x / (-self.spec_min_val) + 1.0  # normalize to [-1, 1]

        l1_loss = F.l1_loss(recon_x, x, reduction="mean")

        # -------- Pearson correlation loss --------
        B = x.size(0)

        x_flat = x.reshape(B, -1)
        recon_flat = recon_x.reshape(B, -1)

        # center
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        recon_centered = recon_flat - recon_flat.mean(dim=1, keepdim=True)

        # compute correlation per sample
        numerator = (x_centered * recon_centered).sum(dim=1)
        denominator = (
            torch.norm(x_centered, dim=1) *
            torch.norm(recon_centered, dim=1)
            + eps
        )

        R = numerator / denominator
        pearson_loss = 1 - R.mean()  # range [0, 2], smaller is better

        # -------- total loss --------
        loss = pearson_loss + alpha * l1_loss

        return loss, {
            "loss": loss.detach(),
            "l1": l1_loss.detach(),
            "pearson": R.mean().detach(),
        }

    def loss_function_simple(self, recon_x, x, eps=1e-8, alpha=0.5):
        """
        normalize x to [-1, 1] by x' = -2 * x / spec_min_val + 1
        perhaps used for planning with only l1 loss
        """
        assert recon_x.dim() == 4 and x.dim() == 4

        x = 2.0 * x / (-self.spec_min_val) + 1.0  # normalize to [-1, 1]

        l1_loss = F.mse_loss(recon_x, x, reduction="mean")

        return l1_loss


class SpectrogramVitDecoderLit(L.LightningModule):

    def __init__(self, low_level_enc, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['low_level_enc'])

        self.cfg = cfg
        self.low_level_enc = low_level_enc
        self.low_level_enc.eval()

        for params in self.low_level_enc.parameters():
            params.requires_grad = False

        self.model = SpectrogramVitDecoder(
            img_size=cfg.model.img_size, patch_size=cfg.model.patch_size,
            input_dim=cfg.model.input_dim, embed_dim=cfg.model.embed_dim, depth=cfg.model.depth,
            num_heads=cfg.model.num_heads, mlp_ratio=cfg.model.mlp_ratio,
            deconv_dim=cfg.model.deconv_dim, out_chans=cfg.model.out_chans,
            norm_layer=nn.LayerNorm, spec_min_val=cfg.model.spec_min_val
        )

        def count_parameters(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            non_trainable = total - trainable
            return total, trainable, non_trainable

        model_total, model_trainable, model_non_trainable = count_parameters(self.model)
        print(f"Model: total={model_total}, trainable={model_trainable}, non-trainable={model_non_trainable}")
        print('----------------------------------------------------------------------------------------------')

    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint["state_dict"]

        drop_keys = [k for k in sd.keys() if k.startswith("low_level_enc.")]
        for k in drop_keys:
            sd.pop(k, None)

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, **kwargs):
        return super().load_from_checkpoint(ckpt_path, strict=False, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = False):
        pruned = {k: v for k, v in state_dict.items() if not k.startswith("low_level_enc.")}
        return super().load_state_dict(pruned, strict=False)

    def _split_batch(self, batch: torch.Tensor):
        spec, _, _ = batch
        spec = spec.unsqueeze(1)
        return spec

    def training_step(self, batch):
        spec = self._split_batch(batch)
        with torch.no_grad():
            z_full = self.low_level_enc(spec)
            z_full = F.layer_norm(z_full, (z_full.size(-1),))

            if z_full.dim() == 3:
                TT, PP = self.model.num_patches
                z_full = z_full.reshape(z_full.size(0), TT, PP, z_full.size(-1))
        recon_x = self.model(z_full)
        loss, logs = self.model.loss_function(recon_x, spec)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in logs.items():
            try:
                self.log(f"{k}", v, on_step=True, on_epoch=True, prog_bar=False)
            except Exception:
                pass

        return loss

    def validation_step(self, batch):
        spec = self._split_batch(batch)
        with torch.no_grad():
            z_full = self.low_level_enc(spec)
            z_full = F.layer_norm(z_full, (z_full.size(-1),))

            if z_full.dim() == 3:
                TT, PP = self.model.num_patches
                z_full = z_full.reshape(z_full.size(0), TT, PP, z_full.size(-1))
        recon_x = self.model(z_full)
        loss, logs = self.model.loss_function(recon_x, spec)
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
