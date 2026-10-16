"""test_loss.py — MAE reconstruction loss contract, per-patch normalization, determinism."""
import torch

import model


def _forward(norm_pix_loss, seed=0, B=2):
    torch.manual_seed(seed)
    m = model.mae_vit_tiny(norm_pix_loss=norm_pix_loss).eval()
    imgs = torch.randn(B, 3, 224, 224)
    with torch.no_grad():
        loss, pred, mask = m(imgs)
    return m, loss, pred, mask


def test_forward_returns_scalar_nonneg_loss():
    # Given a forward pass
    m, loss, pred, mask = _forward(norm_pix_loss=True)

    # Then loss is a non-negative scalar and mask mean ~ mask_ratio
    assert loss.dim() == 0
    assert float(loss) >= 0.0
    assert pred.shape == (2, 3, 224, 224)
    assert mask.shape == (2, m.patch_embed.num_patches)
    assert abs(float(mask.mean()) - 0.75) <= 0.03


def test_norm_pix_loss_on_and_off_are_finite_scalars():
    # Given models built with and without per-patch normalization
    for flag in (True, False):
        m, loss, pred, mask = _forward(norm_pix_loss=flag, seed=1)
        # Then both produce a finite scalar loss
        assert torch.isfinite(loss).item()
        assert loss.dim() == 0
        assert float(loss) >= 0.0


def test_forward_loss_is_zero_when_prediction_matches_target_without_norm():
    # Given a model with raw (un-normalized) pixel targets
    torch.manual_seed(0)
    m = model.mae_vit_tiny(norm_pix_loss=False)
    torch.manual_seed(0)
    imgs = torch.randn(2, 3, 224, 224)
    N = m.patch_embed.num_patches
    mask = torch.ones(2, N)

    # When the prediction equals the input exactly
    loss = m.forward_loss(imgs, imgs.clone(), mask)

    # Then the reconstruction loss is exactly zero
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_forward_loss_only_counts_masked_patches():
    # Given a perfect prediction except for one perturbed patch, and a mask that
    # marks only that patch as "counted".
    torch.manual_seed(0)
    m = model.mae_vit_tiny(norm_pix_loss=False)
    torch.manual_seed(0)
    imgs = torch.randn(1, 3, 224, 224)
    N = m.patch_embed.num_patches
    p = m.patch_size

    target_patch = 3  # an arbitrary patch index
    row, col = divmod(target_patch, m.patch_embed.grid)

    counted = torch.zeros(1, N)
    counted[0, target_patch] = 1.0

    pred = imgs.clone()
    pred[:, :, row * p:(row + 1) * p, col * p:(col + 1) * p] += 5.0

    # When the perturbed patch is counted -> positive loss
    loss_counted = m.forward_loss(imgs, pred, counted)

    # When the perturbed patch is NOT counted -> zero loss
    not_counted = torch.zeros(1, N)
    not_counted[0, (target_patch + 1) % N] = 1.0
    loss_ignored = m.forward_loss(imgs, pred, not_counted)

    assert float(loss_counted) > 0.0
    assert torch.allclose(loss_ignored, torch.tensor(0.0), atol=1e-6)


def test_forward_determinism_with_fixed_seed():
    # Given identical inputs and seeds
    torch.manual_seed(7)
    m = model.mae_vit_tiny().eval()
    imgs = torch.randn(2, 3, 224, 224)

    torch.manual_seed(123)
    with torch.no_grad():
        loss_a, _, _ = m(imgs)
    torch.manual_seed(123)
    with torch.no_grad():
        loss_b, _, _ = m(imgs)

    # Then losses agree
    assert torch.allclose(loss_a, loss_b)
    assert torch.isfinite(loss_a).item()
