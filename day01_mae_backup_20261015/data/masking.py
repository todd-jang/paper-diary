import torch


def patchify(imgs, patch_size=16):
    """
    imgs: (B, 3, H, W)
    return: (B, N, patch_size*patch_size*3), N = (H/p)*(W/p)
    """
    p = patch_size
    B, C, H, W = imgs.shape
    assert H % p == 0 and W % p == 0, "H, W must be divisible by patch_size"
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x)   # (B, h, w, p, p, C)
    x = x.reshape(B, h * w, p * p * C)
    return x


def random_masking(x, mask_ratio=0.75):
    """
    x: (B, N, D)
    return: x_kept (B, N_keep, D), ids_keep, ids_shuffle
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    x_kept = torch.gather(
        x, dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )
    return x_kept, ids_keep, ids_shuffle