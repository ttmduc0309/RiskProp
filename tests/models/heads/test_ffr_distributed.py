from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from taa.models import AnticipationHead


class FFRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs, labels):
        return AnticipationHead._future_frame_regularization_loss(
            inputs * self.weight, [SimpleNamespace(target=x) for x in labels])


def _worker(rank, init_file, labels, queue):
    dist.init_process_group('gloo', init_method='file://' + init_file,
                            rank=rank, world_size=2)
    try:
        model = DistributedDataParallel(FFRModel())
        inputs = torch.tensor([[0., 1., 3.], [0., 2., 5.],
                               [0., 3., 7.], [0., 4., 9.]])
        loss = model(inputs[rank * 2:rank * 2 + 2], labels[rank * 2:rank * 2 + 2])
        loss.backward()
        queue.put((rank, loss.item(), model.module.weight.grad.item()))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('labels', [
    [True, False, True, True],
    [True, True, False, False],
    [False, False, False, False],
    [True, False, True, False],
])
def test_ddp_matches_global_positive_mean(tmp_path, labels):
    inputs = torch.tensor([[0., 1., 3.], [0., 2., 5.],
                           [0., 3., 7.], [0., 4., 9.]])
    reference = FFRModel()
    expected = reference(inputs, labels)
    expected.backward()

    ctx = mp.get_context('spawn')
    queue = ctx.SimpleQueue()
    init_file = str(tmp_path / 'rendezvous')
    mp.spawn(_worker, args=(init_file, labels, queue), nprocs=2, join=True)
    results = [queue.get(), queue.get()]
    for _, _, grad in results:
        assert grad == pytest.approx(reference.weight.grad.item())
    assert sum(loss for _, loss, _ in results) / 2 == pytest.approx(expected.item())
