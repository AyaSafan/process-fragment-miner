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
    export_models_to_pnml,
    export_xes_by_fragments,
    mine_all_fragment_models_and_root,
    remove_activities,
    save_process_trees,
)


def _html_or_print():
    """Return (display_html, print_text) — only one is active:
    HTML in a Jupyter notebook, plain text in a terminal."""
    try:
        from IPython.display import display as _ipyd, HTML as _HTML
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return lambda h: _ipyd(_HTML(h)), lambda t: None
    except Exception:
        pass
    return lambda h: None, lambda t: print(t)


def _fmt_metrics(m):
    return (f'fi={m["fi"]:.4f}  pr={m["pr"]:.4f}  F1={m["F1"]:.4f}  '
            f'CFC={m["CFC"]}  size={m["size"]}')


def evaluation(
    logs_dir,
    export_path,
    path_filtering=False,
    methods=("heuristic", "bigram", "similarity", "frequency"),
    show_fragment_plots=True,
    scorer_kwargs=None,
):
    """
    Runs the full PFM evaluation pipeline.

    1. Loads each XES log from *logs_dir*.
    2. Removes START/END sentinel events (not part of the fragment model).
    3. For each *method* — runs ProcessFragmentMiner with the given scorer to
       discover the best disjoint fragment set.
    4. Mines fragment subprocess models + root abstraction model, computes
       quality metrics (fitness, precision, F1, CFC, size) for each fragment
       and the root model.
    5. Exports per-fragment XES sub-logs, the root abstraction log,
       PNML model files, and metrics to *export_path*.

    Args:
        logs_dir (str): Path to a XES file or a directory of XES files.
        export_path (str): Where to write results (sub-logs go under
            ``{export_path}/xes/``, models under ``{export_path}/pnml/``,
            metrics at ``{export_path}/{log}.pfm.metrics.txt``).
        path_filtering (bool): Keep only the most frequent variants (80 % coverage).
        methods (tuple of str): Scoring methods to evaluate.
        show_fragment_plots (bool): Visualise each fragment's process model
            (default ``True``).  Set to ``False`` to suppress per-fragment plots.
        scorer_kwargs (dict or None): Optional keyword arguments forwarded to
            ProcessFragmentMiner for each method.  For example, the ``"weighted"``
            scorer uses ``scorer_kwargs={"scorers": [("frequency", 0.5), ("heuristic", 0.5)]}``.
            If None, no extra arguments are passed.
    """
    html, txt = _html_or_print()

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

            html(f'<hr><h2>Method: {method} &mdash; score={score:.4f} (algorithm={algorithm_used})</h2>')
            txt(f'\n{"=" * 60}')
            txt(f'  Method: {method}  (score={score:.4f}, algorithm={algorithm_used})')
            txt(f'{"=" * 60}')

            out = mine_all_fragment_models_and_root(
                event_log, fragments,
                include_end_events=True,
                compute_metrics=True,
                show_root_plot=False,
                show_fragment_plots=False,
            )
            (root_log, root_metrics, mean_metrics, _, fragment_trees,
             root_model, fragment_models, fragment_metrics_list) = out
            fragment_trees_by_method[method] = fragment_trees

            for idx, acts in enumerate(fragments):
                html(f'<h3>Fragment {idx} <span style="font-weight:normal;font-size:0.9em">'
                     f'({len(acts)} activities)</span></h3>')
                html(f'<b>Activities:</b> {acts}')
                txt(f'\n  Fragment {idx} ({len(acts)} activities)')
                txt(f'    Activities: {acts}')

                if fragment_metrics_list[idx] is not None:
                    m = fragment_metrics_list[idx]
                    html(f'<br><b>Metrics:</b> {_fmt_metrics(m)}')
                    txt(f'    Metrics: {_fmt_metrics(m)}')

                from process_fragment_miner.utils import visualize_process_model
                visualize_process_model(fragment_models[idx])

            if root_metrics is not None:
                rm = _fmt_metrics(root_metrics)
                html(f'<h3>Root Model</h3><b>Metrics:</b> {rm}')
                txt(f'\n  Root model metrics: {rm}')
            if mean_metrics is not None:
                mm = (f'fi={mean_metrics["fi_mean"]:.4f}  pr={mean_metrics["pr_mean"]:.4f}  '
                      f'F1={mean_metrics["F1_mean"]:.4f}  CFC={mean_metrics["CFC_mean"]:.1f}  '
                      f'size={mean_metrics["size_mean"]:.1f}')
                html(f'<br><b>Mean fragment metrics:</b> {mm}')
                txt(f'  Mean fragment metrics: {mm}')

            export_models_to_pnml(
                root_model, fragment_models, export_path, filename, method,
            )
            export_xes_by_fragments(
                event_log, fragments, export_path, filename, method,
                root_log=root_log,
                include_root=True,
            )

        if len(fragment_trees_by_method) > 0:
            trees_path = scoring_metrics_path.replace('.pfm.metrics.txt', '.process_trees.txt')
            save_process_trees(fragment_trees_by_method, trees_path)
            txt(f'\n  Process trees  -> {trees_path}')
            txt(f'  XES sub-logs   -> {export_path}/xes/')
            txt(f'  PNML models    -> {export_path}/pnml/')
            html(f'<hr><pre>'
                 f'Process trees  -> {trees_path}\n'
                 f'XES sub-logs   -> {export_path}/xes/\n'
                 f'PNML models    -> {export_path}/pnml/'
                 f'</pre>')
