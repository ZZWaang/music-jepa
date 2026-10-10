from __future__ import annotations
import torch
import torch.nn as nn
from ..utils.tensors import trunc_normal_


def expand_size(sz):
    if isinstance(sz, int):
        return [sz, sz]
    return sz


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/
    """
    def __init__(self, patch_size=(2, 8), in_chans=2, embed_dim=384, norm_layer=None, flatten=True, init_std=0.02):
        super().__init__()
        patch_size = expand_size(patch_size)

        # make it compatible with stem transformer
        if len(patch_size) > 2:
            patch_size = patch_size[-2:]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.init_std = init_std
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
        r"""

        Args:
            x (torch.Tensor): input image/LMS, shape (*, in_chans, H, W)

        Returns:
            torch.Tensor: patch-embedded image, shape (*, H / patch_size, W / patch_size, embed_dim)
        """
        *batch_dims, in_chans, height, width = x.size()
        x = self.proj(x.view(-1, in_chans, height, width))
        if self.flatten:
            x = x.permute(0, 2, 3, 1)  # channels-last
        x = self.norm(x)
        return x.view(*batch_dims, *x.shape[-3:])


class PatchEmbed1D(nn.Module):
    """ 1D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/
    """
    def __init__(self, patch_size=8, in_chans=2, embed_dim=384, norm_layer=None, flatten=True, init_std=0.02):
        super().__init__()

        self.flatten = flatten

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x (torch.Tensor): input image/LMS, shape (*, in_chans, H, W)

        Returns:
            torch.Tensor: patch-embedded image, shape (*, H / patch_size, W / patch_size, embed_dim)
        """
        *batch_dims, in_chans, width = x.size()
        x = self.proj(x.view(-1, in_chans, width))
        if self.flatten:
            x = x.permute(0, 2, 1)  # channels-last
        x = self.norm(x)
        return x.view(*batch_dims, *x.shape[-2:])


class PatchUnembed(nn.Module):
    """
    Inverse of PatchEmbed using ConvTranspose2d.
    Input:  (*, H', W', embed_dim)
    Output: (*, out_chans, H'*ph, W'*pw)
    """

    def __init__(self, patch_size=(2, 8), embed_dim=256, deconv_dim=256,
                 out_chans=2, norm_layer=None, init_std=0.02):
        super().__init__()
        ph, pw = expand_size(patch_size)
        self.projT = nn.ConvTranspose2d(embed_dim, deconv_dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.outproj = nn.Conv2d(deconv_dim, out_chans, kernel_size=1)  # to refine output
        self.init_std = init_std
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
        elif isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (*, H', W', D)
        *batch_dims, Hs, Ws, D = x.shape
        x = x.view(-1, Hs, Ws, D)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # -> (N, D, H', W')
        x = self.projT(x)

        x = self.outproj(x)
        return x
