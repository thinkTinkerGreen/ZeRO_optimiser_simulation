"""Communication package for virtual ranks and distributed backends."""

from zerosim.comm.base import CollectiveMismatchError, Comm
from zerosim.comm.dist_comm import DistComm
from zerosim.comm.stats import CommStats
from zerosim.comm.thread_comm import ThreadComm, ThreadCoordinator

__all__ = [
    "Comm",
    "CollectiveMismatchError",
    "CommStats",
    "ThreadComm",
    "ThreadCoordinator",
    "DistComm",
]
