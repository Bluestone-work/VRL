"""Observation-only memory recommendation with learned local interventions."""
import numpy as np
import torch

from marl.temporal_candidate_learning import TemporalCandidateActorCritic
from scripts.tracked_learning_episode import TrackingManeuverLibrary


class MemoryRecommendationLibrary(TrackingManeuverLibrary):
    def prepare(self, packet):
        result = super().prepare(packet)
        features, candidates, valid, details = result
        # Run exactly once per observed control step. This is the same public
        # history-based heuristic used as a comparator, without simulator data.
        self.memory_recommendation = super().conventional_choice(packet, valid, details, memory=True)
        return result

    def conventional_choice(self, packet, valid, details, memory=False):
        if memory: return self.memory_recommendation.copy()
        return np.zeros(self.cfg.clusters, np.int64)


class MemoryPriorActorCritic(TemporalCandidateActorCritic):
    def __init__(self, hidden=128):
        super().__init__(hidden)
        # Approximately 94% nominal probability with ten valid candidates.
        # Deterministic untrained decisions exactly reproduce memory control.
        with torch.no_grad(): self.nominal_bias.fill_(5.)
