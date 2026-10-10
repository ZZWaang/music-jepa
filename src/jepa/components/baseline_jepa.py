import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.ema import _copy_params, _ema_update
from ..utils.statistics import variance_floor, cov_reg_loss
from ..utils.mask import apply_one_mask
import math


class JepaEmaBaselineBase(nn.Module):

    def __init__(self, ema_tau_base=0.95, ema_tau_final=0.95, ema_schedule_steps=50_000,
                 student_final_norm=True, teacher_final_norm=True,
                 loss_type="mse", require_normalize_input=False):
        """
        Base class for baseline audio-only JEPA.
        - This class largely adapted from the JepaEmaBase class.
        """
        super().__init__()

        self.state_student: nn.Module | None = None
        self.state_teacher: nn.Module | None = None

        self.predictor: nn.Module | None = None

        self.student_final_norm = student_final_norm
        self.teacher_final_norm = teacher_final_norm

        assert loss_type in ["mse", "l1"], f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type

        # EMA parameters
        self.register_buffer("ema_tau_base", torch.tensor(float(ema_tau_base)))
        self.register_buffer("ema_tau_final", torch.tensor(float(ema_tau_final)))
        self.ema_schedule_steps = ema_schedule_steps
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.require_normalize_input = False

    def ema_init(self):
        _copy_params(self.state_student, self.state_teacher)
        for p in self.state_teacher.parameters():
            p.requires_grad = False
        self.state_teacher.eval()

    @torch.no_grad()
    def _current_tau(self) -> float:
        if self.ema_schedule_steps is None:
            return float(self.ema_tau_final)

        # cosine schedule from base -> final over ema_schedule_steps
        t = min(int(self.global_step.item()), int(self.ema_schedule_steps))
        cosw = 0.5 * (1 + math.cos(math.pi * t / self.ema_schedule_steps))
        # start near base, end at final
        tau = float(self.ema_tau_final - (self.ema_tau_final - self.ema_tau_base) * cosw)
        return tau

    @torch.no_grad()
    def update_teacher(self):
        tau = self._current_tau()
        _ema_update(self.state_teacher, self.state_student, tau)

    def encode_student_state(self, state, state_pedal, state_context_mask):
        if state_pedal is not None:
            z_mask = self.state_student(state, pedal=state_pedal,
                                        mask=state_context_mask,
                                        require_normalize_input=self.require_normalize_input)
        else:
            z_mask = self.state_student(state, mask=state_context_mask,
                                        require_normalize_input=self.require_normalize_input)

        if self.student_final_norm:
            z_mask = F.layer_norm(z_mask, (z_mask.size(-1),))
        return z_mask

    def encode_student_state_target(self, state, state_pedal, state_prediction_mask):
        with torch.no_grad():
            if state_pedal is not None:
                z_full = self.state_student(state, pedal=state_pedal,
                                            require_normalize_input=self.require_normalize_input)
            else:
                z_full = self.state_student(state, require_normalize_input=self.require_normalize_input)

            if self.student_final_norm:
                z_full = F.layer_norm(z_full, (z_full.size(-1),))

            z_target = apply_one_mask(z_full, state_prediction_mask)

        return z_target

    def encode_teacher_state(self, state, state_pedal, state_prediction_mask):
        with torch.no_grad():
            if state_pedal is not None:
                z_full = self.state_teacher(state, pedal=state_pedal,
                                            require_normalize_input=self.require_normalize_input)
            else:
                z_full = self.state_teacher(state, require_normalize_input=self.require_normalize_input)

            if self.teacher_final_norm:
                z_full = F.layer_norm(z_full, (z_full.size(-1),))

            z_target = apply_one_mask(z_full, state_prediction_mask)

        return z_target, z_full

    def predict(self, z_mask, state_context_mask, state_prediction_mask):
        z_pred = self.predictor(z_mask, state_context_mask, state_prediction_mask)
        return z_pred

    def compute_pred_loss(self, z_pred, z_target, reduce_batch=True):
        loss_func = F.mse_loss if self.loss_type == "mse" else F.l1_loss
        if reduce_batch:
            pred_loss = loss_func(z_pred, z_target)
        else:
            pred_loss = loss_func(z_pred, z_target, reduction="none").mean(dim=(-2, -1))
        return pred_loss

    def compute_vc_statistics(self, z_mask, z_full, z_pred):
        _, std_mask = variance_floor(z_mask)
        _, std_full = variance_floor(z_full)
        cov_loss = cov_reg_loss(z_full)
        _, std_pred = variance_floor(z_pred)
        return std_mask, std_full, std_pred, cov_loss

    def _state_action_pred(self, state: torch.Tensor, state_pedal: torch.Tensor,
                           state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor,
                           reduce_batch=True):
        s_target, s_full = self.encode_teacher_state(state, state_pedal, state_prediction_mask)
        s_mask = self.encode_student_state(state, state_pedal, state_context_mask)
        s_pred = self.predict(s_mask, state_context_mask, state_prediction_mask)
        pred_loss = self.compute_pred_loss(s_pred, s_target, reduce_batch=reduce_batch)
        return pred_loss, s_mask, s_full, s_pred, s_target

    def forward(self, state: torch.Tensor, state_pedal: torch.Tensor,
                state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor):

        pred_loss, s_mask, s_full, s_pred, _ = self._state_action_pred(
            state, state_pedal,
            state_context_mask, state_prediction_mask,
            reduce_batch=True
        )
        s_std_mask, s_std_full, s_std_pred, s_cov_loss = self.compute_vc_statistics(s_mask, s_full, s_pred)
        loss = pred_loss

        with torch.no_grad():
            logs = {
                "010_loss": loss.detach(),
                "011_pred_loss": pred_loss.detach(),
                "021_s_mask_norm": s_mask.norm(dim=-1).mean().detach(),
                "022_s_full_norm": s_full.norm(dim=-1).mean().detach(),
                "023_s_pred_norm": s_pred.norm(dim=-1).mean().detach(),
                "031_s_std_mask": s_std_mask.detach(),
                "032_s_std_full": s_std_full.detach(),
                "033_s_std_pred": s_std_pred.detach(),
                "034_s_cov_loss": s_cov_loss.detach(),
            }
        self.global_step += 1
        return loss, logs

    def _compute_win_rate(self, x_ref, x_comp):
        win_rate = (x_ref < x_comp).float().mean()
        return win_rate

    def eval_batch(self, state: torch.Tensor, state_pedal: torch.Tensor,
                   state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor):
        pred_loss, s_mask, s_full, s_pred, s_target = self._state_action_pred(
            state, state_pedal,
            state_context_mask, state_prediction_mask,
            reduce_batch=False
        )
        loss = pred_loss.mean(-1)
        s_std_mask, s_std_full, s_std_pred, s_cov_loss = self.compute_vc_statistics(s_mask, s_full, s_pred)

        win_rate_dict = {}

        # consider a random state
        B = s_mask.size(0)
        s_mask_random = s_mask[torch.randperm(B)]
        s_pred_random_state = self.predict(s_mask_random, state_context_mask, state_prediction_mask)
        pred_loss_random_state = self.compute_pred_loss(s_pred_random_state, s_target, reduce_batch=False)
        random_state_win_rate = self._compute_win_rate(pred_loss, pred_loss_random_state)
        win_rate_dict["10_wr_st"] = random_state_win_rate.item()

        with torch.no_grad():
            logs = {
                "010_loss": loss.detach(),
                "011_pred_loss": pred_loss.mean(-1).detach(),
                "021_s_mask_norm": s_mask.norm(dim=-1).mean().detach(),
                "022_s_full_norm": s_full.norm(dim=-1).mean().detach(),
                "023_s_pred_norm": s_pred.norm(dim=-1).mean().detach(),
                "031_s_std_mask": s_std_mask.detach(),
                "032_s_std_full": s_std_full.detach(),
                "033_s_std_pred": s_std_pred.detach(),
                "034_s_cov_loss": s_cov_loss.detach(),
            }

        for k, v in win_rate_dict.items():
            logs[k] = v
        return loss, logs
