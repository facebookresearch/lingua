# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import json

import torch

from lingua import checkpoint


def _train_step(model, optimizer):
    optimizer.zero_grad()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()


def _assert_state_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_state_equal(actual_value, expected_value)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected)
    else:
        assert actual == expected


def _get_trained_state():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    _train_step(model, optimizer)
    model_state, optim_state = checkpoint.get_state_dict(model, optimizer)
    return copy.deepcopy(model_state), copy.deepcopy(optim_state)


def test_load_from_checkpoint_applies_model_state(tmp_path, monkeypatch):
    (tmp_path / ".metadata").touch()
    expected_model_state, _ = _get_trained_state()
    model = torch.nn.Linear(3, 2)

    def load(state_dict, checkpoint_id):
        assert checkpoint_id == str(tmp_path)
        state_dict["model"] = copy.deepcopy(expected_model_state)

    monkeypatch.setattr(checkpoint.dcp, "load", load)

    checkpoint.load_from_checkpoint(str(tmp_path), model)

    _assert_state_equal(checkpoint.get_model_state_dict(model), expected_model_state)


def test_checkpoint_manager_applies_model_and_optimizer_state(tmp_path, monkeypatch):
    expected_model_state, expected_optim_state = _get_trained_state()
    checkpoint_path = tmp_path / "0000000001"
    checkpoint_path.mkdir()
    train_state_dict = {"step": 1}
    (checkpoint_path / checkpoint.TRAIN_STATE_NAME.format(0)).write_text(
        json.dumps(train_state_dict)
    )

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    train_state = type(
        "TrainState",
        (),
        {"load_state_dict": lambda self, state: setattr(self, "state", state)},
    )()

    def load(state_dict, checkpoint_id):
        assert checkpoint_id == checkpoint_path
        state_dict["model"] = copy.deepcopy(expected_model_state)
        state_dict["optim"] = copy.deepcopy(expected_optim_state)

    monkeypatch.setattr(checkpoint.dcp, "load", load)

    manager = checkpoint.CheckpointManager(
        checkpoint.CheckpointArgs(path=str(tmp_path))
    )
    manager.load(model, optimizer, train_state, device_mesh=None, path=checkpoint_path)

    actual_model_state, actual_optim_state = checkpoint.get_state_dict(model, optimizer)
    _assert_state_equal(actual_model_state, expected_model_state)
    _assert_state_equal(actual_optim_state, expected_optim_state)
    assert train_state.state == train_state_dict
