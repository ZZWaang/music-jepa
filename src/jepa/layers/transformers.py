from __future__ import annotations
import torch
import torch.nn as nn

from .pos_embed import get_sincos_pos_embed
from .patch_embed import expand_size
from ..utils.tensors import trunc_normal_
from ..utils.mask import apply_one_mask, gather_attn_mask

from.attention import FlashAttentionBlock, FlashAttentionBlockWithAttnMask
from .attention import FlashSelfCrossAttentionBlock


class VisionTransformer(nn.Module):

    """
    ViT for patch embeddings.

    2D positional embeddings + Transformer layers + LayerNorm + optional output projection.
    Patch selection by specifying the mask.
    """

    def __init__(self, img_size=(16, 128), patch_size=(2, 8), embed_dim=384, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02, output_dim=None):
        super().__init__()
        self.img_size, self.patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            FlashAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        if output_dim is not None:
            self.out_proj = nn.Linear(embed_dim, output_dim)
        else:
            self.out_proj = None
        self.init_std = init_std
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        embed_dim = self.pos_embed.size(-1)
        pos_embed = get_sincos_pos_embed(embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, mask=None, return_layers: bool = False):
        # embed patches
        batch_size, T_patches, P_patches, embed_dim = x.size()

        x = x + self.pos_embed[:, :T_patches, :, :]

        # flatten
        x = x.view(batch_size, -1, embed_dim)

        if mask is not None:
            x = apply_one_mask(x, mask)  # (B, n_keep, D)

        # apply Transformer blocks
        layers = []
        for blk in self.blocks:
            x = blk(x)
            if return_layers:
                layers.append(x)

        x = self.norm(x)
        if self.out_proj is not None:
            x = self.out_proj(x)

        if return_layers:
            layers.pop()  # replace the last feature with the normalized one.
            layers.append(x)

        if return_layers:
            return torch.stack(layers)

        return x


class MaskedPredictionTransformer(nn.Module):

    """
    ViT predictor that combines context tokens with learned tokens at masked target positions
    and returns the predicted target tokens.

    2D positional embeddings + Transformer layers + LayerNorm.
    """

    def __init__(self, img_size=(16, 128), patch_size=(2, 8), embed_dim=384, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.img_size, self.patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            FlashAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        embed_dim = self.pos_embed.size(-1)
        pos_embed = get_sincos_pos_embed(embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, context_mask, prediction_mask):
        # embed patches
        if x.dim() == 4:
            B, T, P, D = x.size()
            x = x.reshape(B, T * P, D)

        B, L, D = x.shape
        pos_emb = self.pos_embed.reshape(1, -1, D).repeat(B, 1, 1)

        context_pos_emb = apply_one_mask(pos_emb, context_mask)
        x = x + context_pos_emb

        pred_pos_emb = apply_one_mask(pos_emb, prediction_mask)
        pred_tokens = self.mask_token.repeat(B, pred_pos_emb.size(1), 1)
        pred_tokens = pred_tokens + pred_pos_emb

        x = torch.cat([x, pred_tokens], dim=1)  # (B, L, D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, -pred_tokens.size(1):, :]

        return x


class VisionTransformerWithAttnMask(nn.Module):
    """
    ViT for patch embeddings with attention masking:
    2D positional embeddings + Transformer layers + LayerNorm + optional output projection.
    - mask controls patch selection.
    - attn_mask controls ViT attention.
    - encode_two controls whether to encode two states at a time or not.
      When True, img_size and patch_size are for two states,and attn_mask should be set accordingly.
    """

    def __init__(self, img_size=(16, 128), patch_size=(2, 8), embed_dim=384, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02, output_dim=None, encode_two=False):
        super().__init__()
        self.encode_two = encode_two
        self.img_size, self.patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            FlashAttentionBlockWithAttnMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        if output_dim is not None:
            self.out_proj = nn.Linear(embed_dim, output_dim)
        else:
            self.out_proj = None
        self.init_std = init_std
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        embed_dim = self.pos_embed.size(-1)
        pos_embed = get_sincos_pos_embed(embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        if self.encode_two:
            half = self.pos_embed.size(1) // 2
            self.pos_embed[:, half:].copy_(self.pos_embed[:, :half])

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, mask=None, return_layers: bool = False, attn_mask=None):
        # embed patches
        batch_size, T_patches, P_patches, embed_dim = x.size()

        x = x + self.pos_embed[:, :T_patches, :, :]

        # flatten
        x = x.view(batch_size, -1, embed_dim)

        if mask is not None:
            x = apply_one_mask(x, mask)  # (B, n_keep, D)
            if attn_mask is not None:
                attn_mask = gather_attn_mask(attn_mask, mask)

        # apply Transformer blocks
        layers = []
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
            if return_layers:
                layers.append(x)

        x = self.norm(x)
        if self.out_proj is not None:
            x = self.out_proj(x)

        if return_layers:
            layers.pop()  # replace the last feature with the normalized one.
            layers.append(x)

        if return_layers:
            return torch.stack(layers)

        return x


class ConditionalPredictionTransformer(nn.Module):
    """
    Transformer specifically designed for conditional prediction in Music-JEPA:
    2D positional embeddings + ViT + LayerNorm
    Each ViT block contains self-attention and cross-attention with the condition sequence.

    There are three modes:
    - forward predicts the second half of the sequence (state) from the first half with condition (action).
    - inverse predicts the second half of the sequence (action) from the first half with condition (state).
    - cdae predicts the complete sequence (denoised), conditioned on another sequence (pertubed).
    """

    def __init__(self, img_size=(16, 128), patch_size=(2, 8), embed_dim=384, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02,
                 cond_img_size=None, cond_patch_size=None, mode="inverse"):
        super().__init__()
        self.mode = mode
        assert self.mode in ["inverse", "forward", "cdae"]

        self.img_size, self.patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        if cond_img_size is not None and cond_patch_size is not None:
            self.cond_img_size, self.cond_patch_size = expand_size(cond_img_size), expand_size(cond_patch_size)
            self.cond_grid_size = [s // p for s, p in zip(self.cond_img_size, self.cond_patch_size)]
            self.cond_pos_embed = nn.Parameter(torch.zeros(1, *self.cond_grid_size, embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        else:
            self.cond_pos_embed = None

        self.blocks = nn.ModuleList([
            FlashSelfCrossAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        embed_dim = self.pos_embed.size(-1)
        pos_embed = get_sincos_pos_embed(embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        if self.cond_pos_embed is not None:
            cond_embed_dim = self.cond_pos_embed.size(-1)
            cond_pos_embed = get_sincos_pos_embed(cond_embed_dim, self.cond_grid_size)
            self.cond_pos_embed.data.copy_(torch.from_numpy(cond_pos_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, act):
        # embed patches
        if x.dim() == 4:
            B, T, P, D = x.size()
            x = x.reshape(B, T * P, D)

        B, L, D = x.shape
        pos_emb = self.pos_embed.reshape(1, -1, D).repeat(B, 1, 1)
        if self.mode == "cdae":
            assert pos_emb.size(1) == L
        else:
            assert pos_emb.size(1) == L * 2

        context_pos_emb = pos_emb[:, :L]
        x = x + context_pos_emb

        if self.mode in ["inverse", "forward"]:
            pred_pos_emb = pos_emb[:, L:]
            pred_tokens = self.mask_token.repeat(B, pred_pos_emb.size(1), 1)
            pred_tokens = pred_tokens + pred_pos_emb

            x = torch.cat([x, pred_tokens], dim=1)  # (B, L, D)

        act_pos_emb = self.cond_pos_embed.reshape(1, -1, D).repeat(B, 1, 1)
        if self.mode in ["inverse", "cdae"]:
            assert act.size(1) == act_pos_emb.size(1)
            act = act + act_pos_emb
        else: # self.model == "forward"
            assert act.size(1) * 2 == act_pos_emb.size(1)
            act = act + act_pos_emb[:, act.size(1):]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, act)

        x = self.norm(x)

        if self.mode in ["inverse", "forward"]:
            x = x[:, -pred_tokens.size(1):, :]
        return x


class MaskedPredictionTransformerWithAction(nn.Module):

    """
    ViT predictor that combines context tokens with learned tokens at masked target positions
    and returns the predicted target tokens. Predictor is also conditioned on action.

    2D positional embeddings (on state and action)+ Transformer layers + LayerNorm.
    """

    def __init__(self, img_size=(16, 128), patch_size=(2, 8), embed_dim=384, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02,
                 cond_img_size=None, cond_patch_size=None):
        super().__init__()
        self.img_size, self.patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        if cond_img_size is not None and cond_patch_size is not None:
            self.cond_img_size, self.cond_patch_size = expand_size(cond_img_size), expand_size(cond_patch_size)
            self.cond_grid_size = [s // p for s, p in zip(self.cond_img_size, self.cond_patch_size)]
            self.cond_pos_embed = nn.Parameter(torch.zeros(1, *self.cond_grid_size, embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        else:
            self.cond_pos_embed = None

        self.blocks = nn.ModuleList([
            FlashSelfCrossAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        embed_dim = self.pos_embed.size(-1)
        pos_embed = get_sincos_pos_embed(embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        if self.cond_pos_embed is not None:
            cond_embed_dim = self.cond_pos_embed.size(-1)
            cond_pos_embed = get_sincos_pos_embed(cond_embed_dim, self.cond_grid_size)
            self.cond_pos_embed.data.copy_(torch.from_numpy(cond_pos_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, act, context_mask, prediction_mask, cond_prediction_mask=None):
        # embed patches
        if x.dim() == 4:
            B, T, P, D = x.size()
            x = x.reshape(B, T * P, D)

        B, L, D = x.shape
        pos_emb = self.pos_embed.reshape(1, -1, D).repeat(B, 1, 1)

        context_pos_emb = apply_one_mask(pos_emb, context_mask)
        x = x + context_pos_emb

        pred_pos_emb = apply_one_mask(pos_emb, prediction_mask)
        pred_tokens = self.mask_token.repeat(B, pred_pos_emb.size(1), 1)
        pred_tokens = pred_tokens + pred_pos_emb

        x = torch.cat([x, pred_tokens], dim=1)  # (B, L, D)

        if cond_prediction_mask is not None:
            assert self.cond_pos_embed is not None, "cond_pos_embed must be defined if cond_prediction_mask is provided"
            act_pos_emb = self.cond_pos_embed.reshape(1, -1, D).repeat(B, 1, 1)
            act_pos_emb = apply_one_mask(act_pos_emb, cond_prediction_mask)
            act = act + act_pos_emb

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, act)

        x = self.norm(x)

        x = x[:, -pred_tokens.size(1):, :]
        return x

