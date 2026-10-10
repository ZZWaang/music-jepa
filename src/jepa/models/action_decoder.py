import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import math
import lightning as L
from ..layers.patch_embed import PatchUnembed
from ..layers.transformers import VisionTransformer
from ..layers.attention import AttentivePoolingBlock

def _frame_from_pr(pr_0_127):
    """pr_0_127: (B,1,T,K) -> Ifr: (B,T,K)"""
    return (pr_0_127.squeeze(1) > 0).float()


def _onoff_from_frame(frame):
    """frame: (B,T,K) -> Ion/Ioff: (B,T,K)"""
    prev = torch.zeros_like(frame)
    prev[:, 1:, :] = frame[:, :-1, :]
    ion = ((frame > 0) & (prev == 0)).float()
    ioff = ((frame == 0) & (prev > 0)).float()
    return ion, ioff


def _tri_spread(events, J=5):
    """
    Triangular spreading in frame domain to approximate Kong's regression targets.
    events: (B,T,K) in {0,1}
    returns: (B,T,K) in [0,1]
    """
    if J <= 0:
        return events.clone()

    out = torch.zeros_like(events)
    for d in range(-J, J + 1):
        w = 1.0 - abs(d) / float(J)
        if w <= 0:
            continue
        if d < 0:
            out[:, :d, :] = torch.maximum(out[:, :d, :], events[:, -d:, :] * w)
        elif d > 0:
            out[:, d:, :] = torch.maximum(out[:, d:, :], events[:, :-d, :] * w)
        else:
            out = torch.maximum(out, events)
    return out


def _vel_from_pr(pr_0_127):
    """pr_0_127: (B,1,T,K) -> Ivel: (B,T,K) in [0,1]"""
    return (pr_0_127.squeeze(1) / 127.0).clamp(0.0, 1.0)


def _pedal_targets(pedal_0_127):
    """
    pedal_0_127: (B,1,T) in [0,127]
    returns:
      depth: (B,T) in [0,1]
      onset: (B,T) in {0,1}
      offset:(B,T) in {0,1}
      global_depth: (B,) mean depth
    """
    pedal = pedal_0_127.squeeze(1)  # (B,T)
    depth = (pedal / 127.0).clamp(0.0, 1.0)

    prev = torch.zeros_like(pedal)
    prev[:, 1:] = pedal[:, :-1]
    onset = ((pedal > 0) & (prev == 0)).float()
    offset = ((pedal == 0) & (prev > 0)).float()

    global_depth = depth.mean(dim=1)
    return depth, onset, offset, global_depth


class PianoRollVitDecoderTranscription(nn.Module):
    """
    Outputs:
      note_logits:  (B,4,T,K) channels:
        0 onset_reg (Ron logits, target Gon in [0,1], BCE)
        1 offset_reg (Roff logits, target Goff in [0,1], BCE)
        2 frame      (Pfr logits, target Ifr in {0,1}, BCE)
        3 velocity   (Pvel logits, target Ivel in [0,1], BCE gated by Ion)

      pedal_logits: (B,3,T) channels:
        0 depth      (sigmoid -> [0,1], MSE)
        1 onset      (logits, BCE)
        2 offset     (logits, BCE)

      pedal_global_pred: (B,) mean(depth_pred)
    """

    def __init__(
        self,
        pr_img_size=(400, 88),
        pr_patch_size=(25, 8),
        input_dim=256,
        embed_dim=256,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
        deconv_dim=256,
        norm_layer=nn.LayerNorm,
        lambda_frame=1.0,
        lambda_on=1.0,
        lambda_off=1.0,
        lambda_vel=1.0,
        lambda_pedal=1.0,
        lambda_global=1.0,
        lambda_pedal_on=1.0,
        lambda_pedal_off=1.0,
    ):
        super().__init__()
        self.pr_patch_size = pr_patch_size
        self.pr_img_size = pr_img_size

        self.pr_num_patches = (
            int(math.ceil(pr_img_size[0] / pr_patch_size[0])),
            int(math.ceil(pr_img_size[1] / pr_patch_size[1])),
        )
        self.pr_padded_img_size = (
            self.pr_num_patches[0] * pr_patch_size[0],
            self.pr_num_patches[1] * pr_patch_size[1],
        )

        self.in_linear = nn.Linear(input_dim, embed_dim)

        self.backbone = VisionTransformer(
            img_size=self.pr_padded_img_size,
            patch_size=pr_patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        self.attn_pool = AttentivePoolingBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )

        # NOTES head: 4 channels
        self.note_unembed = PatchUnembed(
            patch_size=pr_patch_size,
            embed_dim=embed_dim,
            deconv_dim=deconv_dim,
            out_chans=4,
            norm_layer=None,
        )

        # PEDAL head: 3 channels (depth, onset, offset)
        self.pedal_unembed = PatchUnembed(
            patch_size=(pr_patch_size[0], 1),
            embed_dim=embed_dim,
            deconv_dim=deconv_dim,
            out_chans=3,
            norm_layer=None,
        )

        # GLOBAL PEDAL head: 1 channel (mean depth)
        self.global_pedal_unembed = PatchUnembed(
            patch_size=(pr_patch_size[0], 1),
            embed_dim=embed_dim,
            deconv_dim=deconv_dim,
            out_chans=1,
            norm_layer=None,
        )

        self.in_linear.apply(self._init_weights)

        self.lambda_frame = lambda_frame
        self.lambda_on = lambda_on
        self.lambda_off = lambda_off
        self.lambda_vel = lambda_vel
        self.lambda_pedal = lambda_pedal
        self.lambda_global = lambda_global
        self.lambda_pedal_on = lambda_pedal_on
        self.lambda_pedal_off = lambda_pedal_off

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        """
        z can be:
          - (B, TT*PP, D) or
          - (B, TT, PP, D)
        returns:
          note_logits: (B,4,T,K)
          pedal_logits:(B,3,T)
          pedal_global:(B,)
        """
        TT, PP = self.pr_num_patches
        if z.dim() == 3:
            z = z.reshape(z.size(0), TT, PP, z.size(-1))  # (B,TT,PP,D)

        B, Hs, Ws, D = z.shape
        h = self.in_linear(z)               # (B,TT,PP,embed)
        h = self.backbone(h)                # (B,TT*PP,embed)
        h = h.view(B, Hs, Ws, -1)           # (B,TT,PP,embed)

        h_pedal = h.reshape(B * TT, PP, -1)                         # (B*TT, PP, embed)
        h_pedal = self.attn_pool(h_pedal).reshape(B, TT, -1).unsqueeze(2)  # (B,TT,1,embed)

        pedal = self.pedal_unembed(h_pedal).squeeze(-1)             # (B,3,padded_T)
        pedal = pedal[:, :, : self.pr_img_size[0]].contiguous()     # (B,3,T)

        note = self.note_unembed(h)                                      # (B,4,padded_T,padded_K)
        note = note[:, :, : self.pr_img_size[0], : self.pr_img_size[1]].contiguous()  # (B,4,T,K)

        pedal_global = self.global_pedal_unembed(h_pedal).squeeze(-1)   # (B,1,padded_T)
        pedal_global = pedal_global[:, :, :self.pr_img_size[0]]             # (B,1,T)
        pedal_global = pedal_global.mean(dim=-1).squeeze(1)  # (B,)

        return note, pedal, pedal_global


    def decode_pr(self, z):
        """
        Returns:
        pr:    (B,1,T,K) in [0,127]
        pedal_depth: (B,1,T)   in [0,127]
        note_matrices: List[Tensor], length B, each (N,4): [midi_pitch, onset_frame_float, offset_frame_float, velocity_0_127]
        """
        note_logits, pedal_logits, _ = self.forward(z)

        # ----- notes -----
        device = note_logits.device
        note_logits = note_logits.cpu()
        note_matrices = self.decode_note_matrix(note_logits) # List[Tensor] length B, each (N,4) with columns [midi_pitch, onset_frame_float, offset_frame_float, velocity_0_127]
        pr = self.note_matrix_to_pr(note_matrices) # (B,1,T,K) in [0,127]
        pr = pr.to(device)
        # ----- pedal -----
        pedal_depth = pedal_logits[:, 0]                                  # (B,T) in [0,1]
        pedal_depth = (pedal_depth.clamp(0.0, 1.0) * 127.0).unsqueeze(1)  # (B,1,T)

        return pr, pedal_depth, note_matrices

    def note_matrix_to_pr(self, note_matrices, T=None, K=None, midi_base=21, dtype=torch.float32):
        """
        Convert per-item note matrices to a velocity pianoroll.

        Args:
        note_matrices: List[Tensor], length B
            each Tensor shape (N,4): [midi_pitch, onset_frame_float, offset_frame_float, velocity_0_127]
            (pitch and velocity are typically integer-valued but stored as floats)
        T: optional int, number of frames (default: self.pr_img_size[0])
        K: optional int, number of pitches (default: self.pr_img_size[1])
        midi_base: MIDI pitch corresponding to k=0 (default 21 for 88-key A0..C8)
        dtype: output dtype

        Returns:
        pr: (B,1,T,K) tensor in [0,127]
        """
        if T is None:
            T = int(self.pr_img_size[0])
        if K is None:
            K = int(self.pr_img_size[1])

        B = len(note_matrices)
        device = note_matrices[0].device if B > 0 and isinstance(note_matrices[0], torch.Tensor) else torch.device("cpu")
        pr = torch.zeros((B, 1, T, K), device=device, dtype=dtype)

        for b in range(B):
            nm = note_matrices[b]
            if nm is None or nm.numel() == 0:
                continue

            # Ensure shape (N,4)
            if nm.dim() != 2 or nm.size(-1) != 4:
                raise ValueError(f"note_matrices[{b}] must have shape (N,4), got {tuple(nm.shape)}")

            # Columns
            pitches = nm[:, 0].to(torch.int64)           # MIDI pitch
            onsets = nm[:, 1].to(torch.float32)
            offsets = nm[:, 2].to(torch.float32)
            vels = nm[:, 3].to(torch.float32).clamp(0.0, 127.0)

            # Map MIDI pitch -> [0, K-1]
            ks = pitches - int(midi_base)

            # Convert float frame times -> integer frame span [start, end)
            starts = torch.floor(onsets).to(torch.int64)
            ends = torch.ceil(offsets).to(torch.int64)

            # Clamp to valid range
            starts = torch.clamp(starts, 0, T)
            ends = torch.clamp(ends, 0, T)

            # Filter invalid notes
            valid = (ks >= 0) & (ks < K) & (ends > starts)
            if valid.sum().item() == 0:
                continue

            ks = ks[valid]
            starts = starts[valid]
            ends = ends[valid]
            vels = vels[valid]

            # Fill pianoroll; overlaps resolved by max
            for i in range(ks.numel()):
                k = int(ks[i].item())
                s = int(starts[i].item())
                e = int(ends[i].item())
                v = float(vels[i].item())
                # max in case of overlaps
                pr[b, 0, s:e, k] = torch.maximum(pr[b, 0, s:e, k], torch.tensor(v, device=device, dtype=dtype))

        return pr

    def decode_note_matrix(self, note_logits, boundary_frame=200):
        """
        Kong-style (local-maximum + sub-frame shift) decoding.

        Inputs:
        note_logits:  (B,4,T,K)  channels: onset, offset, frame, velocity (logits)

        Returns:
        note_matrices: List[torch.Tensor], length B
            each is (N,4) with columns:
            [midi_pitch, onset_frame_float, offset_frame_float, velocity_0_127]
            (pitch & velocity are integer-valued, stored in float tensor for simplicity)
        """
        # ----- notes -----
        note_onsets = torch.sigmoid(note_logits[:, 0])   # (B,T,K)
        note_offsets = torch.sigmoid(note_logits[:, 1])  # (B,T,K)
        note_frames = torch.sigmoid(note_logits[:, 2])   # (B,T,K)
        note_vels = torch.sigmoid(note_logits[:, 3])     # (B,T,K) in [0,1]

        # -------------------- decoding hyperparams (follow Kong's paper) --------------------
        on_th = 0.99     # onset peak threshold
        off_th = 0.90    # offset peak threshold
        fr_th = 0.90     # frame disappearance threshold

        if boundary_frame is not None:
            boundary_mask = (note_frames[:, boundary_frame - 1: -1 : boundary_frame] < fr_th).float()
            for k in range(5):
                note_onsets[:, boundary_frame + k::boundary_frame] *= boundary_mask

        # 88-key mapping: K=88 => MIDI pitches 21..108 (A0..C8)
        midi_base = 21

        B, T, K = note_onsets.shape
        device = note_onsets.device
        eps = 1e-6

        def _refine_shift_abc(x: torch.Tensor, t: int) -> float:
            """
            Kong Eq.(10)/(11) shift in *frame units* using A=t-1, B=t, C=t+1.
            Returns delta in [-0.5, 0.5] approximately.
            """
            if t <= 0 or t >= (x.numel() - 1):
                return 0.0
            yA = float(x[t - 1].item())
            yB = float(x[t].item())
            yC = float(x[t + 1].item())
            # If the peak is flat / degenerate, don't refine.
            if not (yB > max(yA, yC) + 1e-8):
                return 0.0

            if yC >= yA:
                den = max(yB - yA, eps)
                # Eq.(10) with hop=1 frame => shift in frames
                shift = 0.5 * (yC - yA) / den
            else:
                den = max(yB - yC, eps)
                # Eq.(11) with hop=1 frame
                shift = 0.5 * (yA - yC) / den

            # Keep it sane (optional safety)
            if shift > 0.5:
                shift = 0.5
            elif shift < -0.5:
                shift = -0.5
            return float(shift)

        def _find_peaks_with_plateau(x: torch.Tensor, th: float):
            """
            Peak detection that tolerates small plateaus:
            - candidate if x[t] >= th and x[t] >= x[t-1] and x[t] >= x[t+1]
            - if a plateau exists (x[t] == x[t+1] == ...), take its center index
            Returns a python list of peak indices (sorted).
            """
            peaks = []
            t = 1
            # use strict interior; boundaries ignored
            while t < T - 1:
                if x[t] >= th and x[t] >= x[t - 1] and x[t] >= x[t + 1]:
                    # plateau handling
                    l = t
                    r = t
                    while r + 1 < T and float(x[r + 1].item()) == float(x[l].item()):
                        r += 1
                    # ensure plateau is not rising from left or to right
                    left_ok = (l == 0) or (x[l] >= x[l - 1])
                    right_ok = (r == T - 1) or (x[r] >= x[r + 1])
                    if left_ok and right_ok:
                        peak_t = (l + r) // 2
                        peaks.append(int(peak_t))
                    t = r + 1
                else:
                    t += 1
            return peaks

        note_matrices = []

        # -------------------- per-batch, per-pitch decoding --------------------
        for b in range(B):
            notes_b = []

            for k in range(K):
                x_on = note_onsets[b, :, k]   # (T,)
                x_off = note_offsets[b, :, k] # (T,)
                x_fr = note_frames[b, :, k]   # (T,)
                x_vel = note_vels[b, :, k]    # (T,)

                onset_peaks = _find_peaks_with_plateau(x_on, on_th)
                offset_peaks_set = set(_find_peaks_with_plateau(x_off, off_th))

                # process onsets in time order
                for i, t_on in enumerate(onset_peaks):

                    next_on = onset_peaks[i + 1] if (i + 1) < len(onset_peaks) else T

                    # refine onset time (frame units)
                    on_shift = _refine_shift_abc(x_on, t_on)
                    onset_time = float(t_on) + on_shift

                    # velocity at onset frame (Kong: take onset frame)
                    vel_0127 = int(torch.round((x_vel[t_on].clamp(0.0, 1.0) * 127.0)).item())
                    vel_0127 = max(0, min(127, vel_0127))

                    # find offset within (t_on, next_on]
                    t = t_on + 1
                    best_off_peak = None
                    while t < min(next_on, T):
                        # record the first offset peak after onset (could also choose highest)
                        if t in offset_peaks_set and best_off_peak is None:
                            best_off_peak = t

                        # frame disappearance triggers finishing decision (Kong fallback)
                        if x_fr[t] < fr_th:
                            if best_off_peak is not None and best_off_peak <= t:
                                t_off = best_off_peak
                                off_shift = _refine_shift_abc(x_off, t_off)
                            else:
                                t_off = t
                                off_shift = 0.0
                            offset_time = float(t_off) + float(off_shift)
                            break
                        t += 1
                    else:
                        # no frame-disappear found before next onset / end
                        if best_off_peak is not None and best_off_peak < min(next_on, T):
                            t_off = best_off_peak
                            off_shift = _refine_shift_abc(x_off, t_off)
                            offset_time = float(t_off) + float(off_shift)
                        else:
                            # truncate at next onset if it exists, else end of sequence
                            if next_on < T:
                                offset_time = float(next_on)  # truncate
                            else:
                                offset_time = float(T - 1)    # tail

                    # only keep valid notes
                    if offset_time > onset_time + 1e-4:
                        midi_pitch = int(midi_base + k)
                        notes_b.append([float(midi_pitch), onset_time, offset_time, float(vel_0127)])

            if len(notes_b) == 0:
                note_matrices.append(torch.zeros((0, 4), device=device, dtype=torch.float32))
            else:
                note_matrices.append(torch.tensor(notes_b, device=device, dtype=torch.float32))

        return note_matrices

    def loss_function_transcription(
        self,
        note_logits: torch.Tensor,      # (B,4,T,K)
        pedal_logits: torch.Tensor,     # (B,3,T)
        pedal_global: torch.Tensor,     # (B,)
        pr_gt: torch.Tensor,            # (B,1,T,K) 0~127
        pedal_gt: torch.Tensor,         # (B,1,T)   0~127
    ):
        """
        Note loss (Kong 2021):
          lnote = lfr + lon + loff + lvel
          lfr  = sum BCE(Ifr, Pfr)
          lon  = sum BCE(Gon, Ron)
          loff = sum BCE(Goff, Roff)
          lvel = sum Ion * BCE(Ivel, Pvel) / max(1,sum Ion)

        Pedal loss (Fang 2025):
          L = λ1*MSE(depth) + λ2*MSE(global) + λ3*BCE(onset) + λ4*BCE(offset)
        """
        assert pr_gt.dim() == 4 and pr_gt.size(1) == 1
        assert pedal_gt.dim() == 3 and pedal_gt.size(1) == 1

        # ----- note targets -----
        frame = _frame_from_pr(pr_gt)              # Ifr (B,T,K)
        ion, ioff = _onoff_from_frame(frame)       # Ion/Ioff
        ivel = _vel_from_pr(pr_gt)                 # Ivel in [0,1]
        gon = _tri_spread(ion)           # Gon in [0,1]
        goff = _tri_spread(ioff)         # Goff

        ron_logits = note_logits[:, 0]
        roff_logits = note_logits[:, 1]
        pfr_logits = note_logits[:, 2]
        pvel_logits = note_logits[:, 3]

        # rough positive rates of onset / frame labels, used for BCE pos_weight
        p_onset = 0.0014
        p_frame = 0.0188
        pos_w_onset = min((1 - p_onset) / (p_onset + 1e-12), 300.0)  # cap
        pos_w_frame = min((1 - p_frame) / (p_frame + 1e-12), 100.0)

        pos_w_onset_t = torch.tensor(pos_w_onset, device=pfr_logits.device, dtype=pfr_logits.dtype)
        pos_w_frame_t = torch.tensor(pos_w_frame, device=pfr_logits.device, dtype=pfr_logits.dtype)

        l_fr = F.binary_cross_entropy_with_logits(pfr_logits, frame, pos_weight=pos_w_frame_t)
        l_on = F.binary_cross_entropy_with_logits(ron_logits, gon, pos_weight=pos_w_onset_t)
        l_off = F.binary_cross_entropy_with_logits(roff_logits, goff, pos_weight=pos_w_onset_t)

        vel_bce = F.binary_cross_entropy_with_logits(pvel_logits, ivel, reduction="none")  # (B,T,K)
        vel_sum = (vel_bce * ion).sum()
        vel_den = ion.sum().clamp(min=1.0)
        l_vel = vel_sum / vel_den

        # ----- pedal targets -----
        depth_gt, pon_gt, poff_gt, gdepth_gt = _pedal_targets(pedal_gt)

        depth_pred = pedal_logits[:, 0]
        pon_logits = pedal_logits[:, 1]
        poff_logits = pedal_logits[:, 2]
        gdepth_pred = pedal_global

        l_pedal = F.mse_loss(depth_pred, depth_gt)
        l_global = F.mse_loss(gdepth_pred, gdepth_gt)
        l_pon = F.binary_cross_entropy_with_logits(pon_logits, pon_gt)
        l_poff = F.binary_cross_entropy_with_logits(poff_logits, poff_gt)

        loss = (
            self.lambda_frame * l_fr +
            self.lambda_on * l_on +
            self.lambda_off * l_off +
            self.lambda_vel * l_vel +
            self.lambda_pedal * l_pedal +
            self.lambda_global * l_global +
            self.lambda_pedal_on * l_pon +
            self.lambda_pedal_off * l_poff
        )

        logs = {
            "00_loss": loss.detach(),
            "01_fr": l_fr.detach(),
            "02_on": l_on.detach(),
            "03_off": l_off.detach(),
            "04_vel": l_vel.detach(),
            "05_pedal_depth": l_pedal.detach(),
            "06_pedal_global": l_global.detach(),
            "07_pedal_on": l_pon.detach(),
            "08_pedal_off": l_poff.detach(),
        }
        return loss, logs


class PianoRollVitDecoderTranscriptionLit(L.LightningModule):

    def __init__(self, low_level_enc, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['low_level_enc'])

        self.cfg = cfg
        self.low_level_enc = low_level_enc
        self.low_level_enc.eval()
        self.require_normalize_input = False
        self.apply_norm = True

        for params in self.low_level_enc.parameters():
            params.requires_grad = False

        self.model = PianoRollVitDecoderTranscription(
            pr_img_size=cfg.model.pr_img_size, pr_patch_size=cfg.model.pr_patch_size,
            input_dim=cfg.model.input_dim, embed_dim=cfg.model.embed_dim, depth=cfg.model.depth,
            num_heads=cfg.model.num_heads, mlp_ratio=cfg.model.mlp_ratio,
            deconv_dim=cfg.model.deconv_dim,
            norm_layer=nn.LayerNorm,
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
        _, pr, pedal = batch
        pr = pr.unsqueeze(1)
        pedal = pedal.unsqueeze(1)
        return pr, pedal

    def training_step(self, batch):
        pr, pedal = self._split_batch(batch)

        with torch.no_grad():
            z_full = self.low_level_enc(pr, pedal=pedal,
                                        require_normalize_input=self.require_normalize_input)
            if self.apply_norm:
                z_full = F.layer_norm(z_full, (z_full.size(-1),))

            if z_full.dim() == 3:
                TT, PP = self.model.pr_num_patches
                z_full = z_full.reshape(z_full.size(0), TT, PP, z_full.size(-1))

        note_logits, pedal_logits, pedal_global = self.model(z_full)
        loss, logs = self.model.loss_function_transcription(note_logits, pedal_logits, pedal_global, pr, pedal)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in logs.items():
            try:
                self.log(f"{k}", v, on_step=True, on_epoch=True, prog_bar=False)
            except Exception:
                pass

        return loss

    def validation_step(self, batch):
        pr, pedal = self._split_batch(batch)

        with torch.no_grad():
            z_full = self.low_level_enc(pr, pedal=pedal,
                                        require_normalize_input=self.require_normalize_input)
            if self.apply_norm:
                z_full = F.layer_norm(z_full, (z_full.size(-1),))

            if z_full.dim() == 3:
                TT, PP = self.model.pr_num_patches
                z_full = z_full.reshape(z_full.size(0), TT, PP, z_full.size(-1))

        note_logits, pedal_logits, pedal_global = self.model(z_full)
        loss, logs = self.model.loss_function_transcription(note_logits, pedal_logits, pedal_global, pr, pedal)

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
