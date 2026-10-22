"""Unit tests for ThreadComm collective communication and mismatch detection (AC-05)."""

import threading

import torch

from zerosim.comm.base import CollectiveMismatchError
from zerosim.comm.thread_comm import ThreadComm, ThreadCoordinator


def test_thread_comm_all_reduce_mean():
    world_size = 4
    coord = ThreadCoordinator(world_size=world_size, timeout=5.0)
    comms = [ThreadComm(rank=r, coordinator=coord) for r in range(world_size)]
    results: list[torch.Tensor | None] = [None] * world_size

    def worker(rank: int):
        # rank 0: [0, 1], rank 1: [2, 3], rank 2: [4, 5], rank 3: [6, 7]
        t = torch.tensor([rank * 2.0, rank * 2.0 + 1.0], dtype=torch.float64)
        results[rank] = comms[rank].all_reduce_mean(t)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Sum of ranks = [0+2+4+6, 1+3+5+7] = [12.0, 16.0], mean = [3.0, 4.0]
    expected = torch.tensor([3.0, 4.0], dtype=torch.float64)
    for r in range(world_size):
        assert torch.allclose(results[r], expected, atol=1e-12)
        assert comms[r].stats.all_reduce.calls == 1
        assert comms[r].stats.all_reduce.elems == 4  # 2 * numel


def test_thread_comm_reduce_scatter_mean():
    world_size = 4
    coord = ThreadCoordinator(world_size=world_size, timeout=5.0)
    comms = [ThreadComm(rank=r, coordinator=coord) for r in range(world_size)]
    results: list[torch.Tensor | None] = [None] * world_size

    def worker(rank: int):
        # Each rank provides 4 elements: [rank, rank, rank, rank]
        t = torch.full((4,), fill_value=float(rank), dtype=torch.float64)
        results[rank] = comms[rank].reduce_scatter_mean(t)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Mean full vector is [1.5, 1.5, 1.5, 1.5]. Each rank receives 1 element: [1.5]
    expected = torch.tensor([1.5], dtype=torch.float64)
    for r in range(world_size):
        assert torch.allclose(results[r], expected, atol=1e-12)
        assert comms[r].stats.reduce_scatter.calls == 1
        assert comms[r].stats.reduce_scatter.elems == 4


def test_thread_comm_all_gather():
    world_size = 4
    coord = ThreadCoordinator(world_size=world_size, timeout=5.0)
    comms = [ThreadComm(rank=r, coordinator=coord) for r in range(world_size)]
    results: list[torch.Tensor | None] = [None] * world_size

    def worker(rank: int):
        shard = torch.tensor([float(rank * 10)], dtype=torch.float64)
        results[rank] = comms[rank].all_gather(shard)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    expected = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64)
    for r in range(world_size):
        assert torch.allclose(results[r], expected, atol=1e-12)
        assert comms[r].stats.all_gather.calls == 1
        assert comms[r].stats.all_gather.elems == 4


def test_thread_comm_mismatch_detection_raises():
    world_size = 2
    coord = ThreadCoordinator(world_size=world_size, timeout=2.0)
    comms = [ThreadComm(rank=r, coordinator=coord) for r in range(world_size)]
    errors: list[Exception | None] = [None, None]

    def worker(rank: int):
        try:
            if rank == 0:
                t = torch.tensor([1.0, 2.0])
                comms[0].all_reduce_mean(t)
            else:
                # Rank 1 provides different length -> shape mismatch
                t = torch.tensor([1.0, 2.0, 3.0])
                comms[1].all_reduce_mean(t)
        except Exception as e:
            errors[rank] = e

    t0 = threading.Thread(target=worker, args=(0,))
    t1 = threading.Thread(target=worker, args=(1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    assert any(
        isinstance(e, (CollectiveMismatchError, RuntimeError)) for e in errors if e is not None
    )
