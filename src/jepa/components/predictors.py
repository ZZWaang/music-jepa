from __future__ import annotations

import torch.nn as nn

from ..utils.tensors import trunc_normal_
from ..layers.transformers import MaskedPredictionTransformer, \
    ConditionalPredictionTransformer, \
    MaskedPredictionTransformerWithAction


class MaskedPredictor(nn.Module):

    """
    A ViT predictor controlled by context and prediction masks (without extra condition).
    E.g., for baseline audio-only JEPA and action predictor
    """

    def __init__(self, img_size=(16, 128), patch_size=(2, 8), embed_dim=384, depth=6,
                 num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.vit = MaskedPredictionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )

        self.init_std = init_std
        self.pred_norm = norm_layer(embed_dim)
        self.pred_proj = nn.Linear(embed_dim, embed_dim)
        self.pred_norm.apply(self._init_weights)
        self.pred_proj.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, context_mask=None, prediction_mask=None):
        assert x.dim() == 3, "Input x should be of shape (B, L, D)"
        out = self.vit(x, context_mask, prediction_mask)
        out = self.pred_norm(out)
        out = self.pred_proj(out)
        return out

class ConditionalPredictor(nn.Module):

    """
    A ViT predictor taking extra condition, specifically designed for Music-JEPA.
    There are three modes:
    - inverse: predict a_curr from a_prev, s_prev and s_full
    - forward: predict s_curr from s_prev, and a_curr
    - cdae: denoise a_curr
    """

    def __init__(self, img_size=(16, 128), patch_size=(2, 8), embed_dim=384, depth=6,
                 num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02,
                 cond_img_size=None, cond_patch_size=None, mode="inverse"):
        super().__init__()
        self.vit = ConditionalPredictionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            cond_img_size=cond_img_size, cond_patch_size=cond_patch_size, mode=mode
        )

        self.init_std = init_std
        self.pred_norm = norm_layer(embed_dim)
        self.pred_proj = nn.Linear(embed_dim, embed_dim)
        self.pred_norm.apply(self._init_weights)
        self.pred_proj.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, act):
        assert x.dim() == 3, "Input x should be of shape (B, L, D)"
        out = self.vit(x, act)
        out = self.pred_norm(out)
        out = self.pred_proj(out)
        return out


class MaskedPredictorWithAction(nn.Module):

    """
    A ViT predictor controlled by context and prediction masks, taking extra condition (action).
    """

    def __init__(self, img_size=(16, 128), patch_size=(2, 8), embed_dim=384, depth=6,
                 num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02,
                 cond_img_size=None, cond_patch_size=None):
        super().__init__()
        self.vit = MaskedPredictionTransformerWithAction(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            cond_img_size=cond_img_size, cond_patch_size=cond_patch_size
        )

        self.init_std = init_std
        self.pred_norm = norm_layer(embed_dim)
        self.pred_proj = nn.Linear(embed_dim, embed_dim)
        self.pred_norm.apply(self._init_weights)
        self.pred_proj.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, act, context_mask=None, prediction_mask=None, cond_prediction_mask=None):
        assert x.dim() == 3, "Input x should be of shape (B, L, D)"
        out = self.vit(x, act, context_mask, prediction_mask, cond_prediction_mask=cond_prediction_mask)
        out = self.pred_norm(out)
        out = self.pred_proj(out)
        return out
