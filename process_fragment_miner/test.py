"""
End-to-end evaluation pipeline.

Loads event logs, runs ProcessFragmentMiner with all scorers, and
exports fragments to XES + metrics.
"""

import math
from pathlib import Path
import os

from process_fragment_miner import ProcessFragmentMiner
from pm4py.statistics.variants.log.get import get_variants, get_variants_sorted_by_count
from pm4py.algo.filtering.log.variants import variants_filter

from process_fragment_miner.adapters.pm4py_adapter import load_event_log
from process_fragment_miner.utils import (
    export_xes_by_fragments,
    mine_all_fragment_models_and_root,
    remove_activities,
    save_process_trees,
)


def evaluation(
    logs_dir,
    export_path,
    path_filtering=False,
    methods=("heuristic", "bigram", "similarity", "frequency"),
    show_fragment_plots=False,
    split_on_log_move=False,
    scorer_kwargs=None,
):
    """
    Runs the full PFM evaluation pipeline.

    1. Loads each XES log from *logs_dir*.
    2. Removes START/END sentinel events (not part of the fragment model).
    3. For each *method* — runs ProcessFragmentMiner with the given scorer to
       discover the best disjoint fragment set.
    4. Exports per-fragment XES sub-logs, the root abstraction log, and
       metrics to *export_path*.

    Args:
        logs_dir (str): Path to a XES file or a directory of XES files.
        export_path (str): Where to write results (sub-logs go under
            ``{export_path}/xes/``, metrics at ``{export_path}/{log}.pfm.metrics.txt``).
        path_filtering (bool): Keep only the most frequent variants (80 % coverage).
        methods (tuple of str): Scoring methods to evaluate.
        show_fragment_plots (bool): Visualise each fragment's process model.
        split_on_log_move (bool): Passed to projection. If True, split traces
            at non-fragment event boundaries into separate sublogs.
            If False, classic projection (one trace per original trace).
        scorer_kwargs (dict or None): Optional keyword arguments forwarded to
            ProcessFragmentMiner for each method.  For example, the ``"weighted"``
            scorer uses ``scorer_kwargs={"scorers": [("frequency", 0.5), ("heuristic", 0.5)]}``.
            If None, no extra arguments are passed.
    """
    if os.path.isfile(logs_dir):
        filenames = [os.path.basename(logs_dir)]
        logs_dir = os.path.dirname(logs_dir)
    else:
        filenames = [f for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f))]

    Path(f'{export_path}/xes').mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        event_log = load_event_log(os.path.join(logs_dir, filename))
        if path_filtering:
            variants = get_variants(event_log)
            variants_sorted = get_variants_sorted_by_count(variants)
            total_cases = sum(count for _, count in variants_sorted)
            cumulative = 0
            selected_variants = []
            for variant, count in variants_sorted:
                cumulative += count
                selected_variants.append(variant)
                if cumulative / total_cases >= 0.8:
                    break
            event_log = variants_filter.apply(event_log, selected_variants)

        # Remove START/END sentinel events before fragment mining
        event_log_no_sentinels = remove_activities(event_log, ['START', 'END'])

        fragment_trees_by_method = {}
        scoring_metrics_path = f'{export_path}/{filename}.pfm.metrics.txt'

        for method in methods:
            kw = {} if scorer_kwargs is None else scorer_kwargs
            miner = ProcessFragmentMiner(
                event_log=event_log_no_sentinels,
                scorer=method,
                scorer_kwargs=kw,
            )

            # Extract top subtraces from the dependency graph
            subtraces = miner.extract_subtraces(max_depth=1000, min_depth=2, top_k=math.inf)

            # Select best disjoint subset of fragments
            score, fragments, individual_scores, algorithm_used = miner.mine_best_fragments(
                subtraces=subtraces,
                score_agg="sum",
                alpha=0.0,
                beam_size=50,
                max_memory_mb=10000,
                method="auto"
                if filename
                not in [
                    "BPIC15_2f.xes.gz",
                    "BPIC15_3f.xes.gz",
                    "BPIC15_4f.xes.gz",
                    "BPIC15_5f.xes.gz",
                ]
                else "beam",
                return_details=True,
                ensure_coverage=True,
            )

            if score is not None and individual_scores is not None:
                with open(scoring_metrics_path, 'a') as f:
                    f.write(f'{method};{algorithm_used};{score};{fragments}\n')

            print(method)
            root_log, root_metrics, mean_metrics, _, fragment_trees = (
                mine_all_fragment_models_and_root(
                    event_log, fragments,
                    include_end_events=True,
                    compute_metrics=False,
                    show_root_plot=True,
                    show_fragment_plots=show_fragment_plots,
                    split_on_log_move=split_on_log_move,
                )
            )
            fragment_trees_by_method[method] = fragment_trees
            export_xes_by_fragments(
                event_log, fragments, export_path, filename, method,
                root_log=root_log,
                include_root=True,
                split_on_log_move=split_on_log_move,
            )
        
        if len(fragment_trees_by_method) > 0:
            trees_path = scoring_metrics_path.replace('.pfm.metrics.txt', '.process_trees.txt')
            save_process_trees(fragment_trees_by_method, trees_path)
            print(f"Process trees saved to: {trees_path}")
