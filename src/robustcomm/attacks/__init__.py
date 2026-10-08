"""Inter-agent message attacks: perturbation set + FGSM, BPDA, SPSA, Boundary-Pushing, and Camouflage."""
from .base import MessageAttack, PerturbationSet
from .boundary_pushing import BoundaryPushingAttack
from .bpda import BPDAMessageAttack
from .camouflage import CamouflageAttack
from .fgsm import FGSMMessageAttack
from .spsa import SPSAMessageAttack

__all__ = [
    "MessageAttack",
    "PerturbationSet",
    "FGSMMessageAttack",
    "BPDAMessageAttack",
    "SPSAMessageAttack",
    "BoundaryPushingAttack",
    "CamouflageAttack",
]
