import torch
import torch.nn as nn


def _copy_params(src: nn.Module, dst: nn.Module):
    dst.load_state_dict(src.state_dict(), strict=True)


@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, tau: float):
    # params
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(tau).add_(p_s.data, alpha=1.0 - tau)
    # common float buffers (e.g., BatchNorm running stats)
    for (n_t, b_t), (_, b_s) in zip(teacher.named_buffers(), student.named_buffers()):
        if b_t.dtype.is_floating_point:
            b_t.data.mul_(tau).add_(b_s.data, alpha=1.0 - tau)
