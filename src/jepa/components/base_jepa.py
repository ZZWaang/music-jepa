import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.ema import _copy_params, _ema_update
from ..utils.statistics import variance_floor, cov_reg_loss
from ..utils.mask import apply_one_mask
from ..utils.action_transforms import SHIFT_ACTION_CONFIG_SP, SHIFT_ACTION_CONFIG_PR
import math


class JepaEmaBase(nn.Module):

    def __init__(self, ema_tau_base=0.95, ema_tau_final=0.95, ema_schedule_steps=50_000,
                 student_final_norm=True, teacher_final_norm=True,
                 loss_type="mse", train_action_prior=False, random_mask=True, prior_loss_weight=1.0,
                 require_normalize_input=False, action_target_encoder="ema"):
        """
        A base class for Music-JEPA with EMA teacher.

        - A JEPA can be trained in two ways: (1) predict the next state; (2) predict patches at random places.
          The current implementation uses a unified approach for both.
        - This project always uses ema_tau_base == ema_tau_final, so ema_schedule_steps has no effect.
        - Student_final_norm, teacher_final_norm: whether to apply layer norm to the final output of the student/teacher encoder.
        - loss_type: "mse" or "l1"
        - train_action_prior: whether to train the action predictor
        - random_mask: if True, predict masked patch (columns) at random places;
          if False, predict the next state.
        - prior_loss_weight: weight for the action prior loss. Effective only when train_action_prior is True.
        - require_normalize_input: whether to normalize the input state and action to [0, 1] before feeding into the encoder.
        - action_target_encoder: use an "ema" action teacher or detached "student" targets for the action prior.
        """
        super().__init__()

        self.state_student: nn.Module | None = None
        self.state_teacher: nn.Module | None = None

        self.action_student: nn.Module | None = None
        self.action_teacher: nn.Module | None = None

        self.predictor: nn.Module | None = None
        self.prior_predictor: nn.Module | None = None

        self.student_final_norm = student_final_norm
        self.teacher_final_norm = teacher_final_norm

        assert loss_type in ["mse", "l1"], f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type

        self.train_action_prior = train_action_prior
        self.random_mask = random_mask
        self.prior_loss_weight = prior_loss_weight
        self.require_normalize_input = require_normalize_input
        if action_target_encoder not in ("ema", "student"):
            raise ValueError(
                "action_target_encoder must be either 'ema' or 'student', "
                f"got {action_target_encoder!r}."
            )
        self.action_target_encoder = action_target_encoder

        # EMA parameters
        self.register_buffer("ema_tau_base", torch.tensor(float(ema_tau_base)))
        self.register_buffer("ema_tau_final", torch.tensor(float(ema_tau_final)))
        self.ema_schedule_steps = ema_schedule_steps
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

    def ema_init(self):
        _copy_params(self.state_student, self.state_teacher)
        for p in self.state_teacher.parameters():
            p.requires_grad = False
        self.state_teacher.eval()

        if self.action_teacher is not None:
            _copy_params(self.action_student, self.action_teacher)
            for p in self.action_teacher.parameters():
                p.requires_grad = False
            self.action_teacher.eval()
        else:
            assert not self.train_action_prior or self.action_target_encoder == "student"

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
        if self.action_teacher is not None:
            _ema_update(self.action_teacher, self.action_student, tau)
        else:
            assert not self.train_action_prior or self.action_target_encoder == "student"

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

    def encode_student_action(self, action, action_pedal, action_mask):
        if action_pedal is not None:
            z_act = self.action_student(action, pedal=action_pedal,
                                        mask=action_mask,
                                        require_normalize_input=self.require_normalize_input)
        else:
            z_act = self.action_student(action,
                                        mask=action_mask,
                                        require_normalize_input=self.require_normalize_input)

        if self.student_final_norm:
            z_act = F.layer_norm(z_act, (z_act.size(-1),))

        return z_act

    def encode_target_action(self, action, action_pedal, action_mask):
        encoder = (
            self.action_teacher
            if self.action_target_encoder == "ema"
            else self.action_student
        )
        if encoder is None:
            raise RuntimeError("Action targets require an action encoder.")

        with torch.no_grad():
            if action_pedal is not None:
                z_act_full = encoder(
                    action,
                    pedal=action_pedal,
                    require_normalize_input=self.require_normalize_input,
                )
            else:
                z_act_full = encoder(
                    action,
                    require_normalize_input=self.require_normalize_input,
                )

            if self.teacher_final_norm:
                z_act_full = F.layer_norm(z_act_full, (z_act_full.size(-1),))

            z_act_target = apply_one_mask(z_act_full, action_mask)

        return z_act_target, z_act_full

    def predict(self, z_mask, z_act, state_context_mask, state_prediction_mask):
        z_pred = self.predictor(z_mask, z_act, state_context_mask, state_prediction_mask)
        return z_pred

    def prior_predict(self, z_act, action_context_mask, action_prediction_mask):
        a_pred = self.prior_predictor(z_act, action_context_mask, action_prediction_mask)
        return a_pred

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

    def _state_action_pred(self, state: torch.Tensor, action: torch.Tensor, state_pedal: torch.Tensor, action_pedal: torch.Tensor,
                           state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor,
                           action_prediction_mask: torch.Tensor, reduce_batch=True):
        s_target, s_full = self.encode_teacher_state(state, state_pedal, state_prediction_mask)
        s_mask = self.encode_student_state(state, state_pedal, state_context_mask)
        a_mask = self.encode_student_action(action, action_pedal, action_prediction_mask)
        s_pred = self.predict(s_mask, a_mask, state_context_mask, state_prediction_mask)
        pred_loss = self.compute_pred_loss(s_pred, s_target, reduce_batch=reduce_batch)
        return pred_loss, s_mask, s_full, s_pred, s_target, a_mask

    def _action_prior_pred(self, action, action_pedal, action_context_mask, action_prediction_mask, a_mask=None, reduce_batch=True):
        # note that we do this in reversed mask
        if a_mask is None:  # do this part with correct context-prediction relation
            a_mask = self.encode_student_action(action, action_pedal, action_context_mask)
            a_target, a_full = self.encode_target_action(action, action_pedal, action_prediction_mask)
            a_pred = self.prior_predict(a_mask, action_context_mask, action_prediction_mask)
            flipped = False
        else: # do this part with flipped context-prediction relation, because a_mask is computed with flipped masks
            a_target, a_full = self.encode_target_action(action, action_pedal, action_context_mask)
            a_pred = self.prior_predict(a_mask, action_prediction_mask, action_context_mask)
            flipped = True
        prior_loss = self.compute_pred_loss(a_pred, a_target, reduce_batch=reduce_batch)
        return prior_loss, a_mask, a_full, a_pred, a_target, flipped

    def forward(self, state: torch.Tensor, action: torch.Tensor, state_pedal: torch.Tensor, action_pedal: torch.Tensor,
                state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor,
                action_context_mask: torch.Tensor, action_prediction_mask: torch.Tensor):
        if self.train_action_prior:
            return self._forward_with_prior(
                state, action, state_pedal, action_pedal,
                state_context_mask, state_prediction_mask, action_context_mask, action_prediction_mask
            )
        else:
            return self._forward_simple(
                state, action, state_pedal, action_pedal,
                state_context_mask, state_prediction_mask, action_prediction_mask
            )

    def _forward_simple(self, state: torch.Tensor, action: torch.Tensor, state_pedal: torch.Tensor, action_pedal: torch.Tensor,
                        state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor, action_prediction_mask: torch.Tensor):

        pred_loss, s_mask, s_full, s_pred, _, a_mask = self._state_action_pred(
            state, action, state_pedal, action_pedal,
            state_context_mask, state_prediction_mask,
            action_prediction_mask, reduce_batch=True
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
                "024_a_mask_norm": a_mask.norm(dim=-1).mean().detach(),
                "031_s_std_mask": s_std_mask.detach(),
                "032_s_std_full": s_std_full.detach(),
                "033_s_std_pred": s_std_pred.detach(),
                "034_s_cov_loss": s_cov_loss.detach(),
            }
        self.global_step += 1
        return loss, logs

    def _forward_with_prior(self, state: torch.Tensor, action: torch.Tensor, state_pedal: torch.Tensor, action_pedal: torch.Tensor,
                            state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor,
                            action_context_mask: torch.Tensor, action_prediction_mask: torch.Tensor):
        """
        x: (B, T, P)
        Returns: loss (scalar), logs (dict)
        """
        pred_loss, s_mask, s_full, s_pred, _, a_mask = self._state_action_pred(
            state, action, state_pedal, action_pedal,
            state_context_mask, state_prediction_mask,
            action_prediction_mask, reduce_batch=True
        )

        prior_loss, a_mask, a_full, a_pred, _, _ = self._action_prior_pred(
            action, action_pedal, action_context_mask, action_prediction_mask,
            a_mask=a_mask if self.random_mask else None,
            reduce_batch=True
        )

        loss = pred_loss + self.prior_loss_weight * prior_loss

        # compute statistics
        s_std_mask, s_std_full, s_std_pred, s_cov_loss = self.compute_vc_statistics(s_mask, s_full, s_pred)
        a_std_mask, a_std_full, a_std_pred, a_cov_loss = self.compute_vc_statistics(a_mask, a_full, a_pred)

        with torch.no_grad():
            logs = {
                "010_loss": loss.detach(),
                "011_pred_loss": pred_loss.detach(),
                "012_prior_loss": prior_loss.detach(),
                "021_s_mask_norm": s_mask.norm(dim=-1).mean().detach(),
                "022_s_full_norm": s_full.norm(dim=-1).mean().detach(),
                "023_s_pred_norm": s_pred.norm(dim=-1).mean().detach(),
                "024_a_mask_norm": a_mask.norm(dim=-1).mean().detach(),
                "025_a_full_norm": a_full.norm(dim=-1).mean().detach(),
                "026_a_pred_norm": a_pred.norm(dim=-1).mean().detach(),
                "031_s_std_mask": s_std_mask.detach(),
                "032_s_std_full": s_std_full.detach(),
                "033_s_std_pred": s_std_pred.detach(),
                "034_s_cov_loss": s_cov_loss.detach(),
                "035_a_std_mask": a_std_mask.detach(),
                "036_a_std_full": a_std_full.detach(),
                "037_a_std_pred": a_std_pred.detach(),
                "038_a_cov_loss": a_cov_loss.detach(),
            }
        self.global_step += 1
        return loss, logs

    def _compute_win_rate(self, x_ref, x_comp):
        win_rate = (x_ref < x_comp).float().mean()
        return win_rate

    def eval_batch(self, state: torch.Tensor, action: torch.Tensor, state_pedal: torch.Tensor, action_pedal: torch.Tensor,
                   state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor,
                   action_context_mask: torch.Tensor, action_prediction_mask: torch.Tensor):
        if self.train_action_prior:
            return self._eval_batch_with_prior(
                state, action, state_pedal, action_pedal,
                state_context_mask, state_prediction_mask, action_context_mask, action_prediction_mask
            )
        else:
            return self._eval_batch_simple(
                state, action, state_pedal, action_pedal,
                state_context_mask, state_prediction_mask, action_prediction_mask
            )


    def _eval_batch_simple(self, state: torch.Tensor, action: torch.Tensor, state_pedal: torch.Tensor, action_pedal: torch.Tensor,
                           state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor, action_prediction_mask: torch.Tensor):
        pred_loss, s_mask, s_full, s_pred, s_target, a_mask = self._state_action_pred(
            state, action, state_pedal, action_pedal,
            state_context_mask, state_prediction_mask,
            action_prediction_mask, reduce_batch=False
        )
        loss = pred_loss.mean(-1)
        s_std_mask, s_std_full, s_std_pred, s_cov_loss = self.compute_vc_statistics(s_mask, s_full, s_pred)

        win_rate_dict = {}

        # consider a random state
        B = s_mask.size(0)
        s_mask_random = s_mask[torch.randperm(B)]
        s_pred_random_state = self.predict(s_mask_random, a_mask, state_context_mask, state_prediction_mask)
        pred_loss_random_state = self.compute_pred_loss(s_pred_random_state, s_target, reduce_batch=False)
        random_state_win_rate = self._compute_win_rate(pred_loss, pred_loss_random_state)
        win_rate_dict["10_wr_st"] = random_state_win_rate.item()

        # consider a random action
        a_mask_random = a_mask[torch.randperm(B)]
        s_pred_random_act = self.predict(s_mask, a_mask_random, state_context_mask, state_prediction_mask)
        pred_loss_random_act = self.compute_pred_loss(s_pred_random_act, s_target, reduce_batch=False)
        random_action_win_rate = self._compute_win_rate(pred_loss, pred_loss_random_act)
        win_rate_dict["11_wr_act"] = random_action_win_rate.item()

        # consider a shifted action
        shift_config = SHIFT_ACTION_CONFIG_SP if state_pedal is None else SHIFT_ACTION_CONFIG_PR
        shift_type_to_index = {"pshift": "12", "tshift": "13", "scale": "14", "pedal": "15"}
        for shift_type, shift_dict in shift_config.items():
            for shift_val in shift_dict["shift_vals"]:
                shift_func = shift_dict["shift_func"]
                action_shift, action_pedal_shift = shift_func(action, action_pedal, shift_val)
                a_mask_shifted = self.encode_student_action(action_shift, action_pedal_shift, action_prediction_mask)
                s_pred_shifted_act = self.predict(s_mask, a_mask_shifted, state_context_mask, state_prediction_mask)
                pred_loss_shifted_act = self.compute_pred_loss(s_pred_shifted_act, s_target, reduce_batch=False)
                shifted_action_win_rate = self._compute_win_rate(pred_loss, pred_loss_shifted_act)
                abbrev = shift_dict["abbrev_func"](shift_val)
                log_index = shift_type_to_index[shift_type]
                win_rate_dict[f"{log_index}_wr_{abbrev}"] = shifted_action_win_rate.item()

        with torch.no_grad():
            logs = {
                "010_loss": loss.detach(),
                "011_pred_loss": pred_loss.mean(-1).detach(),
                "021_s_mask_norm": s_mask.norm(dim=-1).mean().detach(),
                "022_s_full_norm": s_full.norm(dim=-1).mean().detach(),
                "023_s_pred_norm": s_pred.norm(dim=-1).mean().detach(),
                "024_a_mask_norm": a_mask.norm(dim=-1).mean().detach(),
                "031_s_std_mask": s_std_mask.detach(),
                "032_s_std_full": s_std_full.detach(),
                "033_s_std_pred": s_std_pred.detach(),
                "034_s_cov_loss": s_cov_loss.detach(),
            }

        for k, v in win_rate_dict.items():
            logs[k] = v
        return loss, logs

    def _eval_batch_with_prior(self, state: torch.Tensor, action: torch.Tensor, state_pedal: torch.Tensor, action_pedal: torch.Tensor,
                               state_context_mask: torch.Tensor, state_prediction_mask: torch.Tensor,
                               action_context_mask: torch.Tensor, action_prediction_mask: torch.Tensor):
        pred_loss, s_mask, s_full, s_pred, s_target, a_mask = self._state_action_pred(
            state, action, state_pedal, action_pedal,
            state_context_mask, state_prediction_mask,
            action_prediction_mask, reduce_batch=False
        )

        prior_loss, a_mask, a_full, a_pred, a_target, flipped = self._action_prior_pred(
            action, action_pedal, action_context_mask, action_prediction_mask,
            a_mask=a_mask if self.random_mask else None,
            reduce_batch=False
        )

        loss = pred_loss.mean(-1) + self.prior_loss_weight * prior_loss.mean(-1)

        # compute statistics
        s_std_mask, s_std_full, s_std_pred, s_cov_loss = self.compute_vc_statistics(s_mask, s_full, s_pred)
        a_std_mask, a_std_full, a_std_pred, a_cov_loss = self.compute_vc_statistics(a_mask, a_full, a_pred)

        win_rate_dict = {}

        # consider a random state
        B = s_mask.size(0)
        s_mask_random = s_mask[torch.randperm(B)]
        s_pred_random_state = self.predict(s_mask_random, a_mask, state_context_mask, state_prediction_mask)
        pred_loss_random_state = self.compute_pred_loss(s_pred_random_state, s_target, reduce_batch=False)
        random_state_win_rate = self._compute_win_rate(pred_loss, pred_loss_random_state)
        win_rate_dict["10_wr_st"] = random_state_win_rate.item()

        # consider a random action
        a_mask_random = a_mask[torch.randperm(B)]
        s_pred_random_act = self.predict(s_mask, a_mask_random, state_context_mask, state_prediction_mask)
        pred_loss_random_act = self.compute_pred_loss(s_pred_random_act, s_target, reduce_batch=False)
        random_action_win_rate = self._compute_win_rate(pred_loss, pred_loss_random_act)
        win_rate_dict["11_wr_act"] = random_action_win_rate.item()

        # consider a shifted action
        shift_config = SHIFT_ACTION_CONFIG_SP if state_pedal is None else SHIFT_ACTION_CONFIG_PR
        shift_type_to_index = {"pshift": "12", "tshift": "13", "scale": "14", "pedal": "15"}
        for shift_type, shift_dict in shift_config.items():
            for shift_val in shift_dict["shift_vals"]:
                shift_func = shift_dict["shift_func"]
                action_shift, action_pedal_shift = shift_func(action, action_pedal, shift_val)
                a_mask_shifted = self.encode_student_action(action_shift, action_pedal_shift, action_prediction_mask)
                s_pred_shifted_act = self.predict(s_mask, a_mask_shifted, state_context_mask, state_prediction_mask)
                pred_loss_shifted_act = self.compute_pred_loss(s_pred_shifted_act, s_target, reduce_batch=False)
                shifted_action_win_rate = self._compute_win_rate(pred_loss, pred_loss_shifted_act)
                abbrev = shift_dict["abbrev_func"](shift_val)
                log_index = shift_type_to_index[shift_type]
                win_rate_dict[f"{log_index}_wr_{abbrev}"] = shifted_action_win_rate.item()

        # consider a random action on prior
        B = s_mask.size(0)
        a_mask_random = a_mask[torch.randperm(B)]
        if flipped:
            a_pred_random = self.prior_predict(a_mask_random, action_prediction_mask, action_context_mask)
        else:
            a_pred_random = self.prior_predict(a_mask_random, action_context_mask, action_prediction_mask)

        pred_loss_random_action = self.compute_pred_loss(a_pred_random, a_target, reduce_batch=False)
        random_state_win_rate = self._compute_win_rate(prior_loss, pred_loss_random_action)
        win_rate_dict["20_wr_prior"] = random_state_win_rate.item()

        with torch.no_grad():
            logs = {
                "010_loss": loss.detach(),
                "011_pred_loss": pred_loss.mean(-1).detach(),
                "012_prior_loss": prior_loss.mean(-1).detach(),
                "021_s_mask_norm": s_mask.norm(dim=-1).mean().detach(),
                "022_s_full_norm": s_full.norm(dim=-1).mean().detach(),
                "023_s_pred_norm": s_pred.norm(dim=-1).mean().detach(),
                "024_a_mask_norm": a_mask.norm(dim=-1).mean().detach(),
                "025_a_full_norm": a_full.norm(dim=-1).mean().detach(),
                "026_a_pred_norm": a_pred.norm(dim=-1).mean().detach(),
                "031_s_std_mask": s_std_mask.detach(),
                "032_s_std_full": s_std_full.detach(),
                "033_s_std_pred": s_std_pred.detach(),
                "034_s_cov_loss": s_cov_loss.detach(),
                "035_a_std_mask": a_std_mask.detach(),
                "036_a_std_full": a_std_full.detach(),
                "037_a_std_pred": a_std_pred.detach(),
                "038_a_cov_loss": a_cov_loss.detach(),
            }

        for k, v in win_rate_dict.items():
            logs[k] = v
        return loss, logs
