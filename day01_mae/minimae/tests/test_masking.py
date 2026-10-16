"""test_masking.py — random_masking: ratio, roundtrip, encoder token count."""
import torch

import model


def _model(mask_ratio=0.75):
    torch.manual_seed(0)
    return model.mae_vit_tiny(mask_ratio=mask_ratio)


def test_random_masking_mean_ratio():
    # Given a large batch of tokens and mask_ratio=0.75
    m = _model(mask_ratio=0.75)
    N = m.patch_embed.num_patches
    B = 512
    torch.manual_seed(0)
    x = torch.randn(B, N, 192)

    # When masking
    x_vis, mask, ids_restore = m.random_masking(x)

    # Then the mask mean is ~ mask_ratio and shapes are right
    assert mask.shape == (B, N)
    assert x_vis.shape == (B, int(N * 0.25), 192)
    assert abs(float(mask.mean()) - 0.75) <= 0.03


def test_random_masking_honors_configured_ratio():
    m = _model(mask_ratio=0.5)
    N = m.patch_embed.num_patches
    torch.manual_seed(1)
    x = torch.randn(256, N, 192)
    x_vis, mask, _ = m.random_masking(x)
    assert x_vis.shape[1] == int(N * 0.5)
    assert abs(float(mask.mean()) - 0.5) <= 0.03


def test_ids_restore_inverts_ids_shuffle():
    B, N = 64, 196

    # Given a shuffle and the restore produced by its argsort
    torch.manual_seed(0)
    ids_shuffle = torch.argsort(torch.rand(B, N), dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    identity = torch.arange(N).unsqueeze(0).expand(B, -1)

    # Then restoring a shuffle with its own restore is the identity
    assert torch.equal(torch.gather(ids_shuffle, 1, ids_restore), identity)


def test_model_returned_ids_restore_is_a_valid_permutation():
    m = _model()
    N = m.patch_embed.num_patches
    B = 64
    torch.manual_seed(0)
    x = torch.randn(B, N, 192)

    # When masking
    _, _, ids_restore = m.random_masking(x)

    # Then ids_restore has the right shape and is a permutation of arange(N)
    identity = torch.arange(N).unsqueeze(0).expand(B, -1)
    assert ids_restore.shape == (B, N)
    sorted_restore, _ = torch.sort(ids_restore, dim=1)
    assert torch.equal(sorted_restore, identity)


def test_visible_tokens_are_gathered_from_ids_keep():
    # Given masked tokens
    m = _model(mask_ratio=0.75)
    N = m.patch_embed.num_patches
    B = 8
    torch.manual_seed(3)
    x = torch.randn(B, N, 192)

    # When masking
    x_vis, mask, ids_restore = m.random_masking(x)

    # Then the visible tokens come from the first len_keep shuffled positions
    ids_restore_inv = torch.argsort(ids_restore, dim=1)  # = ids_shuffle
    len_keep = int(N * (1 - m.mask_ratio))
    ids_keep = ids_restore_inv[:, :len_keep]
    expected = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    assert torch.allclose(x_vis, expected)


def test_encoder_receives_expected_token_count():
    # Given ratio 0.75 and N=196 (224/16)
    m = _model(mask_ratio=0.75).eval()
    N = m.patch_embed.num_patches
    assert N == 196
    imgs = torch.randn(2, 3, 224, 224)

    # When running the encoder
    with torch.no_grad():
        latent, mask, _ = m.forward_encoder(imgs)

    # Then only int(N*(1-0.75)) visible tokens are fed to the encoder
    assert latent.shape == (2, int(N * (1 - 0.75)), 192)
    assert latent.shape[1] == 49
    assert mask.shape == (2, N)
