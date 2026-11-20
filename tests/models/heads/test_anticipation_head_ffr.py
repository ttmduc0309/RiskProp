from types import SimpleNamespace

import numpy as np
import torch

from taa.models import AnticipationHead


def _sample(target):
    return SimpleNamespace(target=target)


def test_ffr_uses_only_positive_sequences_and_detaches_future_scores():
    scores = torch.tensor(
        [[0.0, 1.0, 3.0], [4.0, -2.0, 7.0]], requires_grad=True)
    samples = [_sample(True), _sample(False)]

    loss = AnticipationHead._future_frame_regularization_loss(scores, samples)
    assert torch.isclose(loss, torch.tensor(2.5))

    loss.backward()
    expected_grad = torch.tensor([[-1.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
    assert torch.allclose(scores.grad, expected_grad)


def test_ffr_returns_differentiable_zero_for_negative_only_batch():
    scores = torch.tensor(
        [[0.0, 10.0, -5.0], [8.0, -2.0, 3.0]], requires_grad=True)
    samples = [_sample(False), _sample(False)]

    loss = AnticipationHead._future_frame_regularization_loss(scores, samples)
    assert loss.item() == 0.0
    assert loss.requires_grad

    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_ffr_positive_loss_is_independent_of_negative_scores():
    positive_only_scores = torch.tensor([[0.0, 1.0, 3.0]])
    mixed_scores = torch.tensor([[0.0, 1.0, 3.0], [100.0, -50.0, 20.0]])

    positive_loss = AnticipationHead._future_frame_regularization_loss(
        positive_only_scores, [_sample(True)])
    mixed_loss = AnticipationHead._future_frame_regularization_loss(
        mixed_scores, [_sample(True), _sample(False)])

    assert torch.equal(positive_loss, mixed_loss)


def test_ffr_returns_differentiable_zero_for_single_frame_sequence():
    scores = torch.tensor([[2.0]], requires_grad=True)

    loss = AnticipationHead._future_frame_regularization_loss(
        scores, [_sample(True)])
    assert loss.item() == 0.0
    assert loss.requires_grad

    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_constraint_training_loss_path_uses_masked_ffr():
    head = AnticipationHead(
        num_classes=1,
        in_channels=4,
        dropout=0.0,
        label_with="constraint",
        num_clips=3,
    )
    scores = torch.tensor(
        [[0.0, 1.0, 3.0], [4.0, -2.0, 7.0]], requires_grad=True)
    samples = [_sample(True), _sample(False)]

    losses = head.loss_by_feat(scores.reshape(-1), samples)
    # The raw positive-only FFR is 2.5 and the configured weight is 1.5.
    assert torch.isclose(losses["loss_ffr"], torch.tensor(3.75))

    losses["loss_ffr"].backward()
    assert torch.equal(scores.grad[1], torch.zeros_like(scores.grad[1]))


def test_frame_supervision_training_loss_path_uses_masked_ffr():
    head = AnticipationHead(
        num_classes=1,
        in_channels=4,
        dropout=0.0,
        label_with="fix",
        clip_len=5,
        num_clips=3,
    )
    scores = torch.tensor(
        [[0.0, 1.0, 3.0], [4.0, -2.0, 7.0]], requires_grad=True)
    samples = [
        SimpleNamespace(
            target=True,
            frame_inds=np.arange(15),
            clip_len=5,
            accident_frame=14,
            frame_interval=1,
        ),
        SimpleNamespace(
            target=False,
            frame_inds=np.arange(15),
            clip_len=5,
            accident_frame=14,
            frame_interval=1,
        ),
    ]

    losses = head.loss_by_feat(scores.reshape(-1), samples)
    assert torch.isclose(losses["loss_ffr"], torch.tensor(3.75))

    losses["loss_ffr"].backward()
    assert torch.equal(scores.grad[1], torch.zeros_like(scores.grad[1]))
