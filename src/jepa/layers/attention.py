import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import DropPath, Mlp
from ..utils.tensors import trunc_normal_


class FlashAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        # Throughout this project, attn_drop, proj_drop are always set to 0, qkv_bias to True.
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # each one has shape batch_size, num_heads, seq_length, embed_dim

        x = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.attn_drop if self.training else 0.,
                                           ).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # Throughout this project, attn_drop, drop_path are always set to 0.
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FlashCrossAttention(nn.Module):
    """
    Cross-attention using torch.scaled_dot_product_attention.
    q comes from query tensor, kv come from context tensor.

    Shapes:
        q_tokens: (B, Nq, C)
        x_tokens: (B, Nk, C)
        out: (B, Nq, C)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.attn_drop = attn_drop

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_tokens: torch.Tensor, x_tokens: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q_tokens.shape
        B_, Nk, C_ = x_tokens.shape
        assert B == B_ and C == C_, f"Batch/Dim mismatch: q={q_tokens.shape}, x={x_tokens.shape}"

        # project
        q = self.q(q_tokens).reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B,h,Nq,hd)
        kv = self.kv(x_tokens).reshape(B, Nk, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B,h,Nk,hd)

        # attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (B,h,Nq,hd)

        out = out.transpose(1, 2).reshape(B, Nq, C)  # (B,Nq,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FlashCrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_x = norm_layer(dim)

        self.attn = FlashCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q_tokens: torch.Tensor, x_tokens: torch.Tensor) -> torch.Tensor:
        q_tokens = q_tokens + self.drop_path(self.attn(self.norm_q(q_tokens), self.norm_x(x_tokens)))
        q_tokens = q_tokens + self.drop_path(self.mlp(self.norm2(q_tokens)))
        return q_tokens


class AttentivePoolingBlock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., norm_layer=nn.LayerNorm, init_std=0.02,
                 output_dim=None, num_query=1):
        super().__init__()
        self.cross_attn_block = FlashCrossAttentionBlock(
            dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=norm_layer)
        self.num_query = num_query
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query, dim))
        self.norm = norm_layer(dim)
        self.init_std = init_std
        self.initialize_weights()
        self.output_dim = output_dim
        if self.output_dim is not None:
            self.out_proj = nn.Linear(dim, output_dim)
            self.out_norm = norm_layer(output_dim)
            self.out_proj.apply(self._init_weights)
            self.out_norm.apply(self._init_weights)
        else:
            self.out_proj = None
            self.out_norm = None

    def initialize_weights(self):
        trunc_normal_(self.query_tokens, std=self.init_std)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.query_tokens.expand(B, -1, -1)  # (B, num_query, C)
        out = self.cross_attn_block(q, x)  # (B, num_query, C)
        out = self.norm(out)
        if self.out_proj is not None:
            out = self.out_norm(self.out_proj(out))
        return out


class FlashAttentionWithAttnMask(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        # Throughout this project, attn_drop, proj_drop are always set to 0, qkv_bias to True.
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # each one has shape batch_size, num_heads, seq_length, embed_dim

        x = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=attn_mask,
                                           dropout_p=self.attn_drop if self.training else 0.,
                                           ).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttentionBlockWithAttnMask(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # Throughout this project, attn_drop, drop_path are always set to 0.
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttentionWithAttnMask(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FlashSelfCrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        # Throughout this project, attn_drop, drop_path are always set to 0.
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_x = norm_layer(dim)

        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)

        self.attn = FlashAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.cross_attn = FlashCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q_tokens: torch.Tensor, x_tokens: torch.Tensor) -> torch.Tensor:
        q_tokens = q_tokens + self.drop_path(self.attn(self.norm_1(q_tokens)))
        q_tokens = q_tokens + self.drop_path(self.cross_attn(self.norm_q(q_tokens), self.norm_x(x_tokens)))
        q_tokens = q_tokens + self.drop_path(self.mlp(self.norm_2(q_tokens)))
        return q_tokens
