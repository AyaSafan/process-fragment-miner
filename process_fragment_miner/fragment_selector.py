import math
import heapq
import os
from collections import defaultdict
import psutil

def _memory_usage_mb():
    """
    Returns the current memory usage of the process in megabytes.

    Returns:
        float: The memory usage (RSS) in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # in MB

def _convert_to_bitmasked(subtraces):
    """
    Converts a list of (score, trace) tuples into a list where each trace
    is represented by a bitmask, indicating the unique set of events contained in the trace.

    Args:
        subtraces (list): List of (score, trace) tuples, where trace is an iterable of events.

    Returns:
        list: List of (score, trace, mask), where mask is an int bitmask of events.
    """
    all_events = set(e for _, trace in subtraces for e in trace)
    event_to_index = {e: i for i, e in enumerate(all_events)}

    bitmasked = []
    for score, trace in subtraces:
        mask = 0
        for e in trace:
            mask |= 1 << event_to_index[e]
        bitmasked.append((score, trace, mask))
    return bitmasked

def _score_trace_set(scores, bitmask, score_agg, alpha):
    """
    Computes the aggregate score for a set of traces and optionally rewards coverage.

    Args:
        scores (list): List of scores (float) of the chosen traces.
        bitmask (int): Integer bitmask covering the union of event sets in the selected subset.
        score_agg (str): Aggregation strategy: "sum", "mean", or "log_likelihood".
        alpha (float or None): Weight for coverage bonus (number of unique events selected), or None.

    Returns:
        float: The computed aggregate score (with coverage bonus if alpha is given).
    """
    if not scores:
        return float('-inf')

    if score_agg == "sum":
        val = sum(scores)
    elif score_agg == "mean":
        val = sum(scores) / len(scores)
    elif score_agg == "log_likelihood":
        if any(s <= 0 for s in scores):
            return float('-inf')
        val = sum(math.log(s) for s in scores)
    else:
        raise ValueError(f"Unsupported score_agg: {score_agg}")

    if alpha is not None:
        # Coverage bonus: the number of unique events in the subset
        coverage = bin(bitmask).count("1")
        val += alpha * coverage

    return val

def _generate_candidate_state(score, trace, trace_mask, used_mask, trace_list, score_list, score_agg, alpha):
    """
    Attempts to add the given trace to the subset, checking for disjointness.

    Args:
        score (float): The score of the trace being considered.
        trace (iterable): The trace (event sequence) itself.
        trace_mask (int): Bitmask representing the event set for this trace.
        used_mask (int): Bitmask representing already selected events.
        trace_list (list): Current list of traces in the subset.
        score_list (list): Current list of corresponding scores.
        score_agg (str): Aggregation mode ("sum", "mean", etc.).
        alpha (float or None): Coverage bonus scaling, if any.

    Returns:
        tuple or None: (new_mask, new_val, new_traces, new_scores) for the new state,
                       or None if trace overlaps (i.e., it is not disjoint with the current set).
    """
    if used_mask & trace_mask != 0:
        return None  # Not disjoint

    new_mask = used_mask | trace_mask
    new_traces = trace_list + [trace]
    new_scores = score_list + [score]
    new_val = _score_trace_set(new_scores, new_mask, score_agg, alpha)

    return new_mask, new_val, new_traces, new_scores

def _dp_solver(subtraces, score_agg, alpha, return_details, max_memory_mb):
    """
    Dynamic programming solver for finding the best disjoint subset of traces for maximum score.
    State is tracked by a bitmask indicating events already used.

    If memory usage exceeds max_memory_mb, falls back to None (triggers beam search).

    Args:
        subtraces (list): List of (score, trace, mask) tuples.
        score_agg (str): Aggregation mode ("sum", etc.).
        alpha (float or None): Scaling for coverage bonus.
        return_details (bool): Whether to return individual trace scores.
        max_memory_mb (int): Upper runtime memory limit in MB.

    Returns:
        tuple or None: (best_val, best_traces, best_scores) or None if memory exceeded.
    """
    dp = defaultdict(lambda: (float('-inf'), [], []))
    dp[0] = (0, [], [])

    for score, trace, trace_mask in subtraces:
        updates = []
        for used_mask, (curr_val, trace_list, score_list) in dp.items():
            result = _generate_candidate_state(
                score, trace, trace_mask, used_mask,
                trace_list, score_list, score_agg, alpha
            )
            if result:
                new_mask, new_val, new_traces, new_scores = result
                updates.append((new_mask, new_val, new_traces, new_scores))

        for new_mask, new_val, new_traces, new_scores in updates:
            if new_val > dp[new_mask][0]:
                dp[new_mask] = (new_val, new_traces, new_scores)
        
        if _memory_usage_mb() > max_memory_mb:
            return None  # fallback to beam

    best_val, best_traces, best_scores = max(dp.values(), key=lambda x: x[0])
    return (best_val, best_traces, best_scores) if return_details else (best_val, best_traces)

def _beam_solver(subtraces, score_agg, alpha, beam_size, return_details):
    """
    Beam search heuristic for selecting a high-scoring disjoint subset when DP is too costly.

    Args:
        subtraces (list): List of (score, trace, mask) tuples.
        score_agg (str): Aggregation mode.
        alpha (float or None): Scaling for coverage bonus.
        beam_size (int): Beam width.
        return_details (bool): Whether to return individual trace scores.

    Returns:
        tuple: Best found solution as (score, [traces], [scores]) or (score, [traces]).
    """
    # beam items: (neg_val, used_mask, traces, scores, coverage)
    beam = [(0, 0, [], [], 0)]  # Start from empty selection

    for score, trace, trace_mask in subtraces:
        seen = {}
        for neg_val, used_mask, trace_list, score_list, _ in beam:
            result = _generate_candidate_state(
                score, trace, trace_mask, used_mask,
                trace_list, score_list, score_agg, alpha
            )
            if result:
                new_mask, new_val, new_traces, new_scores = result
                tup = (-new_val, new_mask, new_traces, new_scores, bin(new_mask).count("1"))
                if new_mask not in seen or -new_val > seen[new_mask][0]:
                    seen[new_mask] = tup

        combined = list(beam) + list(seen.values())
        beam = heapq.nsmallest(beam_size, combined)

    best = min(beam)
    return (-best[0], best[2], best[3]) if return_details else (-best[0], best[2])

def get_best_disjoint_subset(
    subtraces,
    score_agg="sum",
    alpha=None,
    beam_size=100,
    max_memory_mb=500,
    return_details=False,
    method="auto"
):
    """
    Selects the best subset of disjoint traces to maximize an aggregate score, using either
    dynamic programming (exact, but exponential-time and memory) or beam search (approximate).

    Args:
        subtraces (list): List of (score, trace) tuples; each trace is a list or iterable of events.
        score_agg (str): Aggregation function for scores ("sum", "mean", "log_likelihood").
        alpha (float or None): Optional coefficient for rewarding coverage (number of unique events).
        beam_size (int): Beam width for the beam search algorithm.
        max_memory_mb (int): Max RAM in megabytes for DP solver, else fallback to beam.
        return_details (bool): If True, also return individual trace scores and algorithm name.
        method (str): Which algorithm to use: "dp", "beam", or "auto" (try DP, then fallback).

    Returns:
        tuple: If return_details==False: (total_score, [selected_traces])
               If return_details==True:  (total_score, [selected_traces], [scores], [("dp" or "beam")])

    Raises:
        ValueError: If the method is not recognized.
        RuntimeError: If DP is chosen but memory was insufficient.
    """
    bitmasked = _convert_to_bitmasked(subtraces)

    if method == "dp" or method == "auto":
        result = _dp_solver(bitmasked, score_agg, alpha, return_details, max_memory_mb)
        if result is not None:
            return result + (("dp",) if return_details else ())
        elif method == "dp":
            raise RuntimeError("DP solver failed due to memory constraints.")
        
        print("⚠️  Memory exceeded. Switching to beam search.")

    if method == "beam" or method == "auto":
        return _beam_solver(bitmasked, score_agg, alpha, beam_size, return_details) + (("beam",) if return_details else ())

    raise ValueError(f"Invalid method: {method}. Choose from 'dp', 'beam', or 'auto'.")