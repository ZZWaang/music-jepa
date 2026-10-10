import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.patch_embed import PatchEmbed, PatchEmbed1D
from ..layers.transformers import VisionTransformerWithAttnMask
import math


def build_block_attention_mask(
    q_len: int,
    k_len: int,
    q_block_size: int,
    k_block_size: int,
    mode: str = "block_causal",
    device=None,
) -> torch.Tensor:
    """
    Build a block-structured attention mask for scaled_dot_product_attention.

    Args:
        q_len: length of query sequence
        k_len: length of key sequence
        q_block_size: size of each query block
        k_block_size: size of each key block
        mode:
            - "block_diag": query block i attends only to key block i
            - "block_causal": query block i attends to key blocks <= i
        device: torch device

    Returns:
        mask: (q_len, k_len) bool tensor

    PyTorch scaled_dot_product_attention semantics:
        True  = can attend
        False = cannot attend

    """
    assert q_len > 0 and k_len > 0
    assert q_block_size > 0 and k_block_size > 0
    assert q_len % q_block_size == 0, f"q_len {q_len} must be divisible by q_block_size {q_block_size}"
    assert k_len % k_block_size == 0, f"k_len {k_len} must be divisible by k_block_size {k_block_size}"

    q_idx = torch.arange(q_len, device=device)
    k_idx = torch.arange(k_len, device=device)

    q_block_id = q_idx // q_block_size          # (q_len,)
    k_block_id = k_idx // k_block_size          # (k_len,)

    q_block = q_block_id[:, None]               # (q_len, 1)
    k_block = k_block_id[None, :]               # (1, k_len)

    if mode == "block_diag":
        pattern = (q_block == k_block)

    elif mode == "block_causal":
        pattern = (k_block <= q_block)

    else:
        raise ValueError(f"Unknown mode: {mode}")
    return pattern


class SpVitEncoder(nn.Module):

    def __init__(self, in_chans=1, img_size=(400, 229), patch_size=(25, 15), embed_dim=256, depth=12,
                 num_heads=4, mlp_ratio=2., norm_layer=nn.LayerNorm, spec_min_val=-80., spec_max_val=0.,
                 output_dim=None, encode_two=False):
        """
        encode_two: whether the encoder is used to encode two states (e.g., current and next) in a single forward pass.
            - If True, img_size is the size of two states;
            - If input x has half of the image H, it will be treated as a single state to be encoded.

        Typically, default img_size and patch_size yields 16x16 patches.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encode_two = encode_two

        self.num_patches = (int(math.ceil(img_size[0] / patch_size[0])), int(math.ceil(img_size[1] / patch_size[1])))
        self.padded_img_size = (self.num_patches[0] * patch_size[0], self.num_patches[1] * patch_size[1])
        self.spec_min_val = spec_min_val
        self.spec_max_val = spec_max_val

        self.pad_t, self.pad_f = self.padded_img_size[0] - img_size[0], self.padded_img_size[1] - img_size[1]

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.vit = VisionTransformerWithAttnMask(
            img_size=self.padded_img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            output_dim=output_dim, encode_two=encode_two
        )

    def forward(self, x: torch.Tensor, mask=None, return_layers: bool = False, require_normalize_input=False):
        # x: (B, C, H, W), typically (B, 1, 400, 229)
        if self.pad_t > 0 or self.pad_f > 0:
            x = F.pad(x, (0, self.pad_f, 0, self.pad_t), value=self.spec_min_val)  # Typically, ->(B, 1, 400, 240)
        if require_normalize_input:
            x = (x - self.spec_min_val) / (self.spec_max_val - self.spec_min_val)  # normalize to [0, 1]

        x = self.patch_embed(x)

        # since can encode either one or two states, x.size(1) can either be num_patches[0] or num_patches[0] // 2
        if self.encode_two and x.size(1) == self.num_patches[0]:
            B = x.size(0)
            n_patches = self.num_patches[0] * self.num_patches[1]
            attn_mask = build_block_attention_mask(
                n_patches, n_patches, n_patches // 2, n_patches // 2, mode="block_diag", device=x.device
            )
            attn_mask = attn_mask.repeat(B, 1, 1).unsqueeze(1)
        else:
            attn_mask = None
        return self.vit(x, mask=mask, return_layers=return_layers, attn_mask=attn_mask)  # Typically, (B, 16, 16, D)


class PrVitEncoder(nn.Module):

    def __init__(self, in_chans=1, img_size=(400, 88), patch_size=(25, 6), embed_dim=256, depth=12,
                 num_heads=4, mlp_ratio=2., norm_layer=nn.LayerNorm, pedal_in_chans=1, pr_min_val=0., pr_max_val=127.,
                 output_dim=None, encode_two=False):
        """
        encode_two: whether the encoder is used to encode two states (e.g., current and next) in a single forward pass.
            - If True, img_size is the size of two states;
            - If input x has half of the image H, it will be treated as a single state to be encoded.

        Typically, default img_size and patch_size yields 16x15 patches.
        """
        super().__init__()

        self.patch_size = patch_size  # (25, 6)
        self.encode_two = encode_two
        self.num_patches = (int(math.ceil(img_size[0] / patch_size[0])), int(math.ceil(img_size[1] / patch_size[1])))  # (16, 15)
        self.padded_img_size = (self.num_patches[0] * patch_size[0], self.num_patches[1] * patch_size[1])  # (400, 90)

        self.pad_t, self.pad_f = self.padded_img_size[0] - img_size[0], self.padded_img_size[1] - img_size[1]  # (0, 2)
        self.pr_min_val = pr_min_val
        self.pr_max_val = pr_max_val

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pedal_patch_embed = PatchEmbed1D(patch_size=patch_size[0], in_chans=pedal_in_chans, embed_dim=embed_dim)

        self.vit = VisionTransformerWithAttnMask(
            img_size=self.padded_img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            output_dim=output_dim, encode_two=encode_two
        )

    def forward(self, x: torch.Tensor, pedal: torch.Tensor = None, mask=None,
                return_layers: bool = False, require_normalize_input=False):
        # x: (B, C, H, W), typically (B, 1, 400, 88)
        # pedal: (B, T), typically (B, 400)

        if self.pad_t > 0 or self.pad_f > 0:
            x = F.pad(x, (0, self.pad_f, 0, self.pad_t), value=self.pr_min_val)  # (B, 1, 400, 90)
        if require_normalize_input:
            x = (x - self.pr_min_val) / (self.pr_max_val - self.pr_min_val)  # normalize to [0, 1]
            pedal = (pedal - self.pr_min_val) / (self.pr_max_val - self.pr_min_val)  # normalize to [0, 1]

        x = self.patch_embed(x)
        if pedal is not None:
            pedal_emb = self.pedal_patch_embed(pedal)  # (B, T, D)
            x = x + pedal_emb.unsqueeze(2)  # broadcast to (B, T, P, D)
        else:
            raise ValueError("Pedal input is required for PrVitEncoder.")

        # since can encode either one or two states, x.size(1) can either be num_patches[0] or num_patches[0] // 2
        if self.encode_two and x.size(1) == self.num_patches[0]:
            B = x.size(0)
            n_patches = self.num_patches[0] * self.num_patches[1]
            attn_mask = build_block_attention_mask(
                n_patches, n_patches, n_patches // 2, n_patches // 2, mode="block_diag", device=x.device
            )
            attn_mask = attn_mask.repeat(B, 1, 1).unsqueeze(1)
        else:
            attn_mask = None
        return self.vit(x, mask=mask, return_layers=return_layers, attn_mask=attn_mask)  # (B, 16, 15, D)
