import torch


def apply_one_mask(x, mask):
    mask_keep = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
    return torch.gather(x, dim=1, index=mask_keep)

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep

    """
    all_x = []
    for m in masks:
        all_x.append(apply_one_mask(x, m))
    return torch.cat(all_x, dim=0)


def get_masks(batch, num_patches, ratio):
    B = batch.size(0)
    ratio = ratio[0] + torch.rand(1).item() * (ratio[1] - ratio[0])
    num_keep = int(num_patches * (1. - ratio))
    collated_masks_pred, collated_masks_enc = [], []

    for _ in range(B):
        m = torch.randperm(num_patches)
        collated_masks_enc.append(m[:num_keep])
        collated_masks_pred.append(m[num_keep:])

    collated_masks_enc = torch.stack(collated_masks_enc, dim=0).long()  # [B, num_keep]
    collated_masks_pred = torch.stack(collated_masks_pred, dim=0).long()  # [B, num_pred]
    return batch, collated_masks_enc, collated_masks_pred


def get_fixed_temporal_masks(batch, num_patches_1, num_patches_2, ratio, num_keep=None, device=None, mask_type="random", B=None):
    assert isinstance(num_patches_1, tuple) or isinstance(num_patches_1, list), "num_patches should be a tuple or list of length 2"
    assert isinstance(num_patches_2, tuple) or isinstance(num_patches_2, list), "num_patches should be a tuple or list of length 2"
    assert num_patches_1[0] == num_patches_2[0], "The first dimension of num_patches_1 and num_patches_2 should be the same, representing the temporal dimension"
    assert mask_type in ["random", "temporal_next", "temporal_prev"]

    if batch is not None:
        B = batch.size(0)
    else:
        assert B is not None, "If batch is not provided, B (batch size) must be specified"

    if num_keep is None:
        ratio = ratio[0] + torch.rand(1).item() * (ratio[1] - ratio[0])
        num_keep = int(num_patches_1[0] * (1. - ratio))
    else:
        assert num_keep <= num_patches_1[0]
    collated_masks_pred_1, collated_masks_enc_1 = [], []
    collated_masks_pred_2, collated_masks_enc_2 = [], []

    mask_template_1 = torch.arange(num_patches_1[0] * num_patches_1[1]).reshape(num_patches_1[0], num_patches_1[1])  # (T, P1)
    mask_template_2 = torch.arange(num_patches_2[0] * num_patches_2[1]).reshape(num_patches_2[0], num_patches_2[1])  # (T, P2)

    for _ in range(B):
        if mask_type == "random":
            m = torch.randperm(num_patches_1[0])
        elif mask_type == "temporal_next":
            m = torch.arange(num_patches_1[0])
            m = torch.roll(m, shifts=num_patches_1[0] - num_keep, dims=0)
        elif mask_type == "temporal_prev":
            m = torch.arange(num_patches_1[0])

        mask_enc_1 = mask_template_1[m[:num_keep], :].reshape(-1)  # (num_keep * P1,)
        mask_pred_1 = mask_template_1[m[num_keep:], :].reshape(-1)  # (num_pred * P1,)

        collated_masks_enc_1.append(mask_enc_1)
        collated_masks_pred_1.append(mask_pred_1)

        mask_enc_2 = mask_template_2[m[:num_keep], :].reshape(-1)  # (num_keep * P2,)
        mask_pred_2 = mask_template_2[m[num_keep:], :].reshape(-1)  # (num_pred * P2,)

        collated_masks_enc_2.append(mask_enc_2)
        collated_masks_pred_2.append(mask_pred_2)

    collated_masks_enc_1 = torch.stack(collated_masks_enc_1, dim=0).long()  # [B, num_keep * P1]
    collated_masks_pred_1 = torch.stack(collated_masks_pred_1, dim=0).long()  # [B, num_pred * P1]

    collated_masks_enc_2 = torch.stack(collated_masks_enc_2, dim=0).long()  # [B, num_keep * P2]
    collated_masks_pred_2 = torch.stack(collated_masks_pred_2, dim=0).long()  # [B, num_pred * P2]

    collated_masks_enc_1 = collated_masks_enc_1.to(device).long()
    collated_masks_pred_1 = collated_masks_pred_1.to(device).long()
    collated_masks_enc_2 = collated_masks_enc_2.to(device).long()
    collated_masks_pred_2 = collated_masks_pred_2.to(device).long()

    return collated_masks_enc_1, collated_masks_pred_1, collated_masks_enc_2, collated_masks_pred_2


def gather_attn_mask(attn_mask, index):
    """
    attn_mask: (B, 1, L, L)
    index:     (B, K)   # token indices to keep

    return:    (B, 1, K, K)
    """
    B, H, L1, L2 = attn_mask.shape
    assert H == 1, f"Expected second dim = 1, got {H}"
    assert L1 == L2, f"Expected square attention mask, got {L1} x {L2}"
    assert index.dim() == 2 and index.shape[0] == B

    K = index.shape[1]

    # First gather rows: (B, 1, K, L)
    row_index = index[:, None, :, None].expand(B, 1, K, L1)
    out = torch.gather(attn_mask, dim=2, index=row_index)

    # Then gather columns: (B, 1, K, K)
    col_index = index[:, None, None, :].expand(B, 1, K, K)
    out = torch.gather(out, dim=3, index=col_index)

    return out
