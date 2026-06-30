from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from process_fragment_miner.adapters.word2vec_adapter import Word2VecAdapter
from process_fragment_miner.utils import project_log_to_activities


'''
Scoring strategies
- bigram — Laplace-smoothed bigram probability (how likely is this activity sequence given training data?)
- dependency — product of dependency strengths along the path
- similarity — average Word2Vec cosine similarity between consecutive activities
- frequency — how many sublog traces result from projecting the full log onto the fragment
'''

class BaseScorer:
    def score(self, trace: List[str]) -> float:
        raise NotImplementedError("Must implement score(trace)")


class BigramScorer(BaseScorer):
    """
    Scores traces based on bigram likelihoods derived from training traces.
    """

    def __init__(self, traces: List[List[str]], smoothing: float = 1.0):
        """
        Args:
            traces (List[List[str]]): Training traces
            smoothing (float): Laplace smoothing factor
        """
        self.smoothing = smoothing
        self.unigrams, self.bigrams, self.total_unigrams = self._build_ngram_counts(traces)

    def score(self, trace: List[str]) -> float:
        """
        Scores a single trace using smoothed bigram likelihood.
        """
        score = self._compute_trace_likelihood(trace)
        return score if score is not None else float('-inf')

    def _build_ngram_counts(self, traces: List[List[str]]):
        unigram_counts = defaultdict(int)
        bigram_counts = defaultdict(int)
        total_unigrams = 0

        for trace in traces:
            for i, word in enumerate(trace):
                unigram_counts[word] += 1
                total_unigrams += 1
                if i > 0:
                    bigram_counts[(trace[i - 1], word)] += 1

        return unigram_counts, bigram_counts, total_unigrams

    def _compute_trace_likelihood(self, trace: List[str]) -> float:
        vocab_size = len(self.unigrams)
        likelihood = 1.0

        for i, word in enumerate(trace):
            if i == 0:
                # Probability of first word (unigram)
                p = (self.unigrams[word] + self.smoothing) / (self.total_unigrams + self.smoothing * vocab_size)
            else:
                prev_word = trace[i - 1]
                p = (self.bigrams[(prev_word, word)] + self.smoothing) / \
                    (self.unigrams[prev_word] + self.smoothing * vocab_size)
            likelihood *= p

        return likelihood

class DependencyScorer(BaseScorer):
    """
    Scores traces based on dependency matrix values between activities.
    """
    def __init__(self, dependency_matrix: Dict[str, Dict[str, float]]):
        self.matrix = dependency_matrix

    def score(self, trace: List[str]) -> float:
        if len(trace) < 2:
            return float('-inf')
        score = 1.0
        for i in range(len(trace) - 1):
            a, b = trace[i], trace[i + 1]
            dep = self.matrix.get(a, {}).get(b, None)
            if dep is None:
                return float('-inf')
            score *= dep
        return score

class SimilarityScorer(BaseScorer):
    """
    Scores traces based on average pairwise Word2Vec similarity between activities.
    """

    def __init__(self, traces: List[List[str]], remove_loops: bool = False):
        """
        Initializes the scorer using a Word2VecAdapter trained on traces.

        Args:
            train_traces: List of training traces
            remove_loops: Whether to exclude traces with loops in bulk scoring
        """
        self.model = Word2VecAdapter(traces)
        self.remove_loops = remove_loops

    def score(self, trace: List[str]) -> float:
        """
        Computes the average similarity of consecutive pairs in a trace.

        Args:
            trace: A list of activity labels

        Returns:
            float similarity score or -inf if any activity is OOV
        """
        if len(trace) < 2:
            return 0.0

        try:
            similarities = [
                self.model.similarity(trace[i], trace[i + 1])
                for i in range(len(trace) - 1)
            ]
            return float(np.mean(similarities))
        except KeyError:
            return float('-inf')

    def score_traces(self, traces: List[List[str]]) -> List[Tuple[List[str], float]]:
        """
        Scores and ranks a list of traces.

        Returns:
            List of (trace, score) sorted descending
        """
        def has_loops(trace):
            return len(set(trace)) < len(trace)

        if self.remove_loops:
            traces = [t for t in traces if not has_loops(t)]

        seen = set()
        unique_traces = []
        for t in traces:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                unique_traces.append(t)

        scored = [(t, self.score(t)) for t in unique_traces]
        return sorted(scored, key=lambda x: x[1], reverse=True)


class FrequencyScorer(BaseScorer):
    """
    Scores a fragment by how many sublog traces result from projecting the
    full event log onto the fragment's activities.  More frequent fragments
    (those whose activities appear in more traces / more blocks) get higher
    scores.
    """

    def __init__(self, event_log):
        self._event_log = event_log
        self._cache = {}

    def score(self, trace: List[str]) -> float:
        key = frozenset(trace)
        if key not in self._cache:
            projected = project_log_to_activities(self._event_log, trace)
            self._cache[key] = len(projected)
        return float(self._cache[key])


class NormalizedWeightedScorer(BaseScorer):
    """
    Meta-scorer that combines multiple sub-scorers via weighted min-max
    normalization.  Each sub-scorer is fit on the full event-log traces to
    determine its scoring range, and individual scores are normalised to
    [0, 1] before being combined as a weighted sum.

    The *scorers* parameter accepts a list of ``(name_or_instance, weight)``
    tuples.  *name_or_instance* can be a string (one of the known scorer names)
    or an already-instantiated ``BaseScorer``.  Typical usage::

        ProcessFragmentMiner(
            event_log=log,
            scorer="weighted",
            scorer_kwargs={
                "scorers": [("heuristic", 0.5), ("frequency", 0.5)],
            },
        )
    """

    def __init__(
        self,
        event_log,
        traces: List[List[str]],
        dependencies: Dict[str, Dict[str, float]],
        scorers: List[Tuple] = None,
    ):
        """
        Args:
            event_log: Full PM4Py EventLog (passed by ScorerFactory from the miner).
            traces: All full traces as List[List[str]] (from miner.get_traces()).
            dependencies: Dependency matrix (from miner.dependencies).
            scorers: List of (name_or_instance, weight) tuples.
                     String names are resolved internally; existing BaseScorer
                     instances are used as-is.
        """
        if scorers is None:
            scorers = [("heuristic", 1.0)]

        # Resolve string names to BaseScorer instances.
        self._sub_scorers = []
        self._weights = []
        for name, weight in scorers:
            self._weights.append(weight)
            if isinstance(name, BaseScorer):
                self._sub_scorers.append(name)
            else:
                self._sub_scorers.append(self._resolve_sub_scorer(name, event_log, traces, dependencies))

        total_w = sum(abs(w) for w in self._weights)
        if total_w > 0:
            self._weights = [w / total_w for w in self._weights]

        # Fit: score every full trace with every sub-scorer to obtain
        # per-scorer min/max for min-max normalisation.
        self._mins = [float("inf")] * len(self._sub_scorers)
        self._maxs = [float("-inf")] * len(self._sub_scorers)
        self._fitted = False
        self._fit(traces)

    # ------------------------------------------------------------------
    # Internal sub-scorer resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_sub_scorer(name, event_log, traces, dependencies):
        """Resolve a scorer name to a BaseScorer instance."""
        name = name.lower()
        if name == "heuristic":
            return DependencyScorer(dependency_matrix=dependencies)
        elif name == "dependency":
            return DependencyScorer(dependency_matrix=dependencies)
        elif name == "bigram":
            return BigramScorer(traces=traces)
        elif name == "similarity":
            return SimilarityScorer(traces=traces)
        elif name == "frequency":
            return FrequencyScorer(event_log=event_log)
        else:
            raise ValueError(
                f"Unknown sub-scorer '{name}' in NormalizedWeightedScorer. "
                f"Known: heuristic, bigram, similarity, frequency"
            )

    # ------------------------------------------------------------------
    # Fit: compute per-scorer min/max on the full trace set
    # ------------------------------------------------------------------
    def _fit(self, traces: List[List[str]]):
        """Compute normalisation bounds by scoring every full trace."""
        for i, scorer in enumerate(self._sub_scorers):
            raw_scores = []
            for trace in traces:
                s = scorer.score(trace)
                if s != float("-inf"):
                    raw_scores.append(s)
            if raw_scores:
                self._mins[i] = min(raw_scores)
                self._maxs[i] = max(raw_scores)
            else:
                self._mins[i] = 0.0
                self._maxs[i] = 1.0
        self._fitted = True

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _min_max_normalize(value, lo, hi):
        """Min-max normalise *value* to [0, 1] given observed extents *lo*, *hi*."""
        if hi <= lo:
            return 0.5
        clipped = max(lo, min(value, hi))
        return (clipped - lo) / (hi - lo)

    def score(self, trace: List[str]) -> float:
        """
        Return the weighted sum of min-max normalised sub-scorer scores.

        Args:
            trace: A list of activity labels to score.

        Returns:
            float in [-inf, inf], typically in [0, 1] per sub-scorer range.
        """
        total = 0.0
        for scorer, weight, mn, mx in zip(
            self._sub_scorers, self._weights, self._mins, self._maxs
        ):
            raw = scorer.score(trace)
            if raw == float("-inf"):
                return float("-inf")
            normalised = self._min_max_normalize(raw, mn, mx)
            total += weight * normalised
        return total