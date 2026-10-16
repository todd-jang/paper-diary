"""test_utils.py — seeding, device selection, MetricWriter CSV, merge_config precedence."""
import csv
import os

import torch

import utils


def test_seed_everything_is_reproducible():
    # Given a fixed seed applied twice
    utils.seed_everything(0)
    a = torch.rand(3)
    utils.seed_everything(0)
    b = torch.rand(3)

    # Then the random draws are identical
    assert torch.equal(a, b)


def test_seed_everything_different_seeds_differ():
    utils.seed_everything(1)
    a = torch.rand(3)
    utils.seed_everything(2)
    b = torch.rand(3)
    assert not torch.equal(a, b)


def test_get_device_explicit_preference_is_returned_unchanged():
    # CPU only — never touch mps/cuda in tests.
    assert utils.get_device("cpu") == "cpu"


def test_metric_writer_writes_header_then_row(tmp_path):
    # Given a fresh metrics file
    path = os.path.join(tmp_path, "metrics.csv")
    writer = utils.MetricWriter(path, ["epoch", "loss"])

    # Then the header row exists immediately after construction
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["epoch", "loss"]]

    # When logging a data row
    writer.log(epoch=1, loss=0.5)

    # Then the file contains header + one data row, in column order
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["epoch", "loss"]
    assert rows[1] == ["1", "0.5"]
    assert len(rows) == 2


def test_metric_writer_missing_fields_become_empty(tmp_path):
    path = os.path.join(tmp_path, "metrics.csv")
    writer = utils.MetricWriter(path, ["epoch", "loss", "lr"])
    writer.log(epoch=2)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1] == ["2", "", ""]


def test_merge_config_precedence_cli_over_config_over_defaults():
    defaults = {"a": 1, "b": 2, "c": 3}
    config = {"b": 20, "c": 30, "z": 999}
    cli = {"c": 300, "b": None}

    out = utils.merge_config(defaults, config, cli)

    # CLI (non-None) > config > defaults; unknown keys are dropped.
    assert out["a"] == 1        # from defaults
    assert out["b"] == 20       # CLI None does not override -> config wins
    assert out["c"] == 300      # CLI wins
    assert "z" not in out       # not a known key
    assert set(out.keys()) == {"a", "b", "c"}


def test_load_config_json(tmp_path):
    import json

    path = os.path.join(tmp_path, "cfg.json")
    with open(path, "w") as f:
        json.dump({"img_size": 224, "mask_ratio": 0.75}, f)

    cfg = utils.load_config(path)
    assert cfg == {"img_size": 224, "mask_ratio": 0.75}
    assert utils.load_config(None) == {}
