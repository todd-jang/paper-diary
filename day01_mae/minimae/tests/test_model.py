"""test_model.py — patchify/unpatchify, positional embeddings, params, forward shapes."""
import numpy as np
import torch

import model


def test_patchify_unpatchify_roundtrip():
    # Given random images (B,3,224,224)
    m = model.mae_vit_tiny()
    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224)

    # When patchifying then unpatchifying
    patches = m.patchify(x)
    recon = m.unpatchify(patches)

    # Then the roundtrip is (near) identity and shapes are correct
    assert recon.shape == x.shape
    assert torch.allclose(recon, x, atol=1e-5)


def test_patchify_shape_and_token_layout():
    m = model.mae_vit_tiny()
    g = m.patch_embed.grid
    p = m.patch_size
    torch.manual_seed(1)
    x = torch.randn(2, 3, 224, 224)

    patches = m.patchify(x)
    # (B, num_patches, patch*patch*channels)
    assert patches.shape == (2, g * g, p * p * 3)
    assert patches.shape[1] == m.patch_embed.num_patches == 196


def test_sincos_pos_embed_shape_determinism_and_bounds():
    embed_dim = 192
    grid = 14
    N = grid * grid

    # When computing the 2D sin-cos embedding twice
    pe1 = model.get_2d_sincos_pos_embed(embed_dim, grid)
    pe2 = model.get_2d_sincos_pos_embed(embed_dim, grid)

    # Then shape is (num_patches, embed_dim), deterministic, and bounded in [-1, 1]
    assert pe1.shape == (N, embed_dim)
    assert np.array_equal(pe1, pe2)
    assert pe1.min() >= -1.0 - 1e-6
    assert pe1.max() <= 1.0 + 1e-6


def test_model_registered_pos_embed_shapes():
    m = model.mae_vit_tiny()
    N = m.patch_embed.num_patches
    # Buffers carry a leading batch dim of 1; the embedding itself is (N, D).
    assert m.enc_pos_embed.squeeze(0).shape == (N, 192)
    assert m.dec_pos_embed.squeeze(0).shape == (N, 128)


def test_forward_output_shapes_and_dtypes():
    # Given a small batch and mask_ratio 0.75
    torch.manual_seed(0)
    m = model.mae_vit_tiny(mask_ratio=0.75).eval()
    imgs = torch.randn(2, 3, 224, 224)

    # When forwarding
    with torch.no_grad():
        loss, pred, mask = m(imgs)

    # Then shapes match the MAE contract
    assert m.patch_embed.num_patches == 196
    assert pred.shape == (2, 3, 224, 224)
    assert mask.shape == (2, m.patch_embed.num_patches)
    assert mask.shape == (2, 196)
    assert loss.dim() == 0


def test_param_count_sanity():
    # Given the ViT-Tiny preset
    m = model.mae_vit_tiny()
    total = sum(p.numel() for p in m.parameters())

    # Then the parameter count is in the expected ~3.3-3.4M range
    assert 3.0e6 <= total <= 3.6e6


def test_model_is_deterministic_with_seed():
    # Given the same input and seed reset between calls
    torch.manual_seed(0)
    m = model.mae_vit_tiny().eval()
    imgs = torch.randn(2, 3, 224, 224)

    torch.manual_seed(42)
    with torch.no_grad():
        loss_a, pred_a, mask_a = m(imgs)
    torch.manual_seed(42)
    with torch.no_grad():
        loss_b, pred_b, mask_b = m(imgs)

    # Then both runs are identical
    assert torch.allclose(loss_a, loss_b)
    assert torch.allclose(pred_a, pred_b)
    assert torch.equal(mask_a, mask_b)
