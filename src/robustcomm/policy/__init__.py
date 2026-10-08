"""Surrogate policies and MADDPG-style ensembles."""
from .surrogate_policy import SurrogateObjective, LinearSurrogatePolicy
from .ensembles import LinearPolicy, PolicyEnsemble

__all__ = ["SurrogateObjective", "LinearSurrogatePolicy", "LinearPolicy", "PolicyEnsemble"]
