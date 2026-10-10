import torch
import torch.nn.functional as F


def variance_floor(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4, after_norm: bool = False):
    """
    Penalize low per-dimension batch std.
    z: (B, D) latent from the STUDENT encoder.
    gamma: desired minimum std per dim. If after_norm=True (unit-sphere), for D=256 use ~0.06.
    """
    zf = z.float()
    if after_norm:
        zf = F.normalize(zf, dim=-1, eps=1e-6)          # angular spread on the sphere
    zf = zf - zf.mean(dim=0, keepdim=True)              # center per dim
    std = torch.sqrt(zf.var(dim=0, unbiased=False) + eps)  # (D,)
    vloss = torch.relu(gamma - std).mean()              # penalize dims with std < gamma
    return vloss, std.mean()


def cov_reg_loss(Z: torch.Tensor, eps: float = 1e-5, normalize_by_dim: bool = True):
    """
    Z: (N, D) features, one row per sample. Flatten tokens first: Z.reshape(N * T, D).
    """
    # center
    Z = Z.reshape(-1, Z.shape[-1])
    Z = Z - Z.mean(dim=0, keepdim=True)
    N = Z.shape[0]

    # covariance: (D, D)
    C = (Z.T @ Z) / (max(N - 1, 1))

    # off-diagonal squared sum
    D = C.shape[0]
    off_diag = C - torch.diag(torch.diag(C))
    loss = (off_diag ** 2).sum()

    if normalize_by_dim:
        loss = loss / D

    return loss
