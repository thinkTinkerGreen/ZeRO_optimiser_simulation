"""Multi-process distributed parity test (E6) comparing DistComm to ThreadComm (AC-13)."""

import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from zerosim.comm.dist_comm import DistComm
from zerosim.config import MODEL_TINY, POLICY_FP32, SimConfig
from zerosim.data import generate_batch, get_rank_batch
from zerosim.ledger import MemoryLedger
from zerosim.runner import get_stage_cls, run_simulation


def _dist_worker(
    rank: int,
    world_size: int,
    stage_idx: int,
    steps: int,
    return_dict: dict,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29505"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.set_num_threads(1)

    cfg = SimConfig(
        world_size=world_size,
        stage=stage_idx,
        policy=POLICY_FP32,
        model=MODEL_TINY,
        microbatch=2,
        steps=steps,
        compute_dtype="float64",
        seed=1234,
    )

    stage_cls = get_stage_cls(stage_idx)
    comm = DistComm(comm_bytes_per_elem=cfg.policy.comm_param_bytes)
    ledger = MemoryLedger(rank=rank)
    stage = stage_cls(cfg=cfg, rank=rank, comm=comm, ledger=ledger, governor=None)

    losses = []
    for step_idx in range(steps):
        inputs, targets = generate_batch(cfg, step=step_idx, device="cpu")
        r_inp, r_tar = get_rank_batch(inputs, targets, rank=rank, microbatch=cfg.microbatch)
        step_loss = stage.step(step_idx, r_inp, r_tar)
        losses.append(step_loss)

    final_params = stage.get_full_parameters()
    return_dict[rank] = {
        "losses": losses,
        "final_params": final_params.detach().cpu(),
    }
    dist.destroy_process_group()


@pytest.mark.parametrize("stage_idx", [0, 1, 2, 3])
def test_e6_dist_comm_vs_thread_comm_parity(stage_idx: int):
    world_size = 4
    steps = 4
    cfg = SimConfig(
        world_size=world_size,
        stage=stage_idx,
        policy=POLICY_FP32,
        model=MODEL_TINY,
        microbatch=2,
        steps=steps,
        compute_dtype="float64",
        seed=1234,
    )

    # 1. Run simulation via ThreadComm
    thread_res = run_simulation(cfg)

    # 2. Run real multi-process distributed training via Gloo and DistComm
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        _dist_worker,
        args=(world_size, stage_idx, steps, return_dict),
        nprocs=world_size,
        join=True,
    )

    # Average losses across dist ranks
    dist_mean_losses = []
    for s in range(steps):
        dist_s = sum(return_dict[r]["losses"][s] for r in range(world_size)) / float(world_size)
        dist_mean_losses.append(dist_s)

    # Assert loss agreement within 1e-12 absolute
    for s in range(steps):
        loss_diff = abs(thread_res.losses_per_step[s] - dist_mean_losses[s])
        assert (
            loss_diff <= 1e-12
        ), f"Stage {stage_idx} step {s} ThreadComm vs DistComm diff {loss_diff} > 1e-12"

    # Assert parameter agreement within 1e-12 absolute
    dist_p0 = return_dict[0]["final_params"]
    param_diff = (thread_res.final_parameters.cpu() - dist_p0).abs().max().item()
    assert (
        param_diff <= 1e-12
    ), f"Stage {stage_idx} param diff {param_diff} > 1e-12 between ThreadComm and DistComm"
