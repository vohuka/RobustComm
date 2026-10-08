"""Data layer: schemas, topologies, synthetic generators, real-env adapters."""
from .schemas import Topology, CommRound, AggregationResult, EvalRecord
from .topologies import build_topology
from .synthetic import generate_message_rounds
from .adapters import make_environment, EnvAdapter, DIALEnvAdapter, MPEEnvAdapter, SMACEnvAdapter

__all__ = [
    "Topology", "CommRound", "AggregationResult", "EvalRecord", "build_topology",
    "generate_message_rounds", "make_environment", "EnvAdapter",
    "DIALEnvAdapter", "MPEEnvAdapter", "SMACEnvAdapter",
]
