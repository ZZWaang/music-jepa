import torch


SHIFT_ACTION_CONFIG_SP = {
    "pshift": {
        "shift_vals": [-6, -4, 4, 6],
        "shift_func": lambda x, ped, shift: (shift_x(x, shift, shift_type="pshift", pad_value=0), ped),
        "abbrev_func": lambda shift: f"p{shift}" if shift < 0 else f"p+{shift}"
        },
    "tshift": {
        "shift_vals": [-10, -5, 5, 10],
        "shift_func": lambda x, ped, shift: (shift_x(x, shift, shift_type="tshift", pad_value=0), ped),
        "abbrev_func": lambda shift: f"t{shift}" if shift < 0 else f"t+{shift}"
    },
    "scale": {
        "shift_vals": [0.4, 0.7, 1.3, 1.6],
        "shift_func": lambda x, ped, scale: (scale_x(x, scale, min_val=0, max_val=127, mode="affine"), ped),
        "abbrev_func": lambda scale: f"pr*{scale}"
    },
    "pedal": {
        "shift_vals": [0.4, 0.7, 1.3, 1.6],
        "shift_func": lambda x, ped, scale: (x, scale_x(ped, scale, min_val=0, max_val=127, mode="affine")),
        "abbrev_func": lambda scale: f"pd*{scale}"
    }
}

SHIFT_ACTION_CONFIG_PR = {
    "pshift": {
        "shift_vals": [-6, -4, 4, 6],
        "shift_func": lambda x, _, shift: (shift_x(x, shift, shift_type="pshift", pad_value=0), None),
        "abbrev_func": lambda shift: f"p{shift}" if shift < 0 else f"p+{shift}"
    },
    "tshift": {
        "shift_vals": [-10, -5, 5, 10],
        "shift_func": lambda x, _, shift: (shift_x(x, shift, shift_type="tshift", pad_value=0), None),
        "abbrev_func": lambda shift: f"t{shift}" if shift < 0 else f"t+{shift}"
    },
    "scale": {"shift_vals": [0.4, 0.7, 1.3, 1.6],
              "shift_func": lambda x, _, scale: (scale_x(x, scale, min_val=-80.0, max_val=0.0, mode="affine"), None),
               "abbrev_func": lambda scale: f"sp*{scale}"
              }
}


def shift_x(x, shift, shift_type="pshift", pad_value=0):
    assert shift_type in ["pshift", "tshift"]

    dim = -1 if shift_type == "pshift" else -2

    y = torch.ones_like(x) * pad_value

    if shift > 0:
        src = [slice(None)] * x.ndim
        dst = [slice(None)] * x.ndim
        src[dim] = slice(0, -shift)
        dst[dim] = slice(shift, None)

        y[tuple(dst)] = x[tuple(src)]
    elif shift < 0:
        src = [slice(None)] * x.ndim
        dst = [slice(None)] * x.ndim
        src[dim] = slice(-shift, None)
        dst[dim] = slice(0, shift)

        y[tuple(dst)] = x[tuple(src)]
    else:
        return x.clone()
    return y


def scale_x(
    x,
    scale,
    min_val,
    max_val,
    chans=None,
    mode="affine",          # "affine" | "multiply"
    ):
    """
    x: Tensor [B, C, ...]
    scale: float or tensor broadcastable to x
    min_val, max_val: scalar
    chans: int or list[int] or None
    mode:
        - "affine": min_val + scale * (x - min_val)
        - "multiply": x * scale
    nonzero_only:
        - True: only scale non-zero entries (for pianoroll / pedal)
    """

    x = x.clone()

    # ---- select channels ----
    if chans is None:
        x_sel = x
    else:
        x_sel = x[:, chans]

    # ---- scaling ----
    if mode == "affine":
        x_sel = min_val + scale * (x_sel - min_val)
    elif mode == "multiply":
        x_sel = x_sel * scale
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- clamp ----
    x_sel = torch.clamp(x_sel, min_val, max_val)

    # ---- write back ----
    if chans is None:
        x = x_sel
    else:
        x[:, chans] = x_sel

    return x
