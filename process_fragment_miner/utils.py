from collections import defaultdict
from copy import deepcopy
import os
import re
import pandas as pd

import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.convert import convert_to_bpmn
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.discovery import discover_process_tree_inductive, discover_heuristics_net
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.variants.log.get import get_variants, get_variants_sorted_by_count
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.util import sorting
from pm4py.objects.conversion.log import converter as log_converter


# ---------------------------------------------------------------------------
#  Log merging
# ---------------------------------------------------------------------------

def merge_event_logs_by_trace_id(log_1, log_2):
    """
    Merges two PM4Py EventLog objects based on common trace IDs.
    Keeps only common columns, removes NaNs, and ensures 'START' is first in each trace.

    Args:
        log_1: First PM4Py EventLog.
        log_2: Second PM4Py EventLog.

    Returns:
        Merged PM4Py EventLog.
    """
    df1 = log_converter.apply(log_1, variant=log_converter.Variants.TO_DATA_FRAME)
    df2 = log_converter.apply(log_2, variant=log_converter.Variants.TO_DATA_FRAME)

    common_columns = df1.columns.intersection(df2.columns)
    df1 = df1[common_columns].copy()
    df2 = df2[common_columns].copy()

    combined_df = pd.concat([df1, df2], ignore_index=True)

    required = {"case:concept:name", "time:timestamp", "concept:name"}
    if not required.issubset(combined_df.columns):
        raise ValueError(f"Missing one or more required columns: {required}")

    combined_df["start_priority"] = combined_df["concept:name"].apply(lambda x: 0 if x == "START" else 1)
    combined_df.sort_values(by=["case:concept:name", "start_priority", "time:timestamp"], inplace=True)
    combined_df.drop(columns=["start_priority"], inplace=True)
    combined_df.dropna(inplace=True)

    return log_converter.apply(combined_df, variant=log_converter.Variants.TO_EVENT_LOG)


# ---------------------------------------------------------------------------
#  Quality metrics (fitness, precision, F1, CFC, size)
# ---------------------------------------------------------------------------

def calculate_cfc(petri_net):
    """
    Control Flow Complexity of a Petri net — counts routing places/transitions.

    Args:
        petri_net: A PM4Py PetriNet object.

    Returns:
        int: CFC value.
    """
    cfc = 0
    for place in petri_net.places:
        if len(place.in_arcs) > 1 or len(place.out_arcs) > 1:
            cfc += 1
    for transition in petri_net.transitions:
        if len(transition.in_arcs) > 1 or len(transition.out_arcs) > 1:
            cfc += 1
    return cfc


def calculate_metrics(process_tree, event_log):
    """
    Computes fitness, precision, F1, CFC, and model size for a process tree
    with respect to an event log.

    Args:
        process_tree: PM4Py ProcessTree object.
        event_log: PM4Py EventLog object.

    Returns:
        dict: {"fi" (fitness), "pr" (precision), "F1", "CFC", "size"}.
    """
    net, initial_marking, final_marking = pt_converter.apply(process_tree)
    log_fitness = pm4py.fitness_alignments(event_log, net, initial_marking, final_marking, multi_processing=False)['log_fitness']
    precision = pm4py.precision_alignments(event_log, net, initial_marking, final_marking, multi_processing=False)

    denominator = log_fitness + precision
    f1 = 2 * (log_fitness * precision) / denominator if denominator != 0 else 0.0

    return {
        'fi':  log_fitness,
        'pr':  precision,
        'F1':  f1,
        'CFC': calculate_cfc(net),
        'size': len(net.places) + len(net.transitions),
    }


def calculate_mean_quality_measures(fragment_properties_list):
    """
    Averages quality metrics across multiple fragments.

    Args:
        fragment_properties_list: List of dicts, each containing a 'metrics' key
                                  per fragment (nested as {group_name: {metrics: ...}}).

    Returns:
        dict: Mean values per metric, keys suffixed with '_mean'.
    """
    sums = defaultdict(float)
    count = 0

    for entry in fragment_properties_list:
        inner = list(entry.values())[0]
        metrics = inner.get('metrics', {})
        for key, value in metrics.items():
            sums[key] += value
        count += 1

    return {f'{k}_mean': v / count for k, v in sums.items()}


# ---------------------------------------------------------------------------
#  Event log projection — reducing the log to one fragment's activities
# ---------------------------------------------------------------------------

def project_log_to_activities(event_log, activity_names):
    """
    Reduces an event log to only the traces/events whose activity name
    appears in *activity_names* (the **projection** step for a fragment).

    Conceptually:  L ↦ L↓_{A_f}   (keep only events whose label ∈ A_f)

    Args:
        event_log: PM4Py EventLog.
        activity_names (list of str): Activity labels to keep.

    Returns:
        PM4Py EventLog containing only events with matching labels.
    """
    return attributes_filter.apply_events(
        event_log,
        parameters={"attribute_key": "concept:name", "positive": True},
        values=activity_names,
    )


def remove_activities(event_log, activity_names):
    """
    Drops all events whose activity name appears in *activity_names*.

    Args:
        event_log: PM4Py EventLog.
        activity_names (list of str): Activity labels to remove.

    Returns:
        PM4Py EventLog with those events removed.
    """
    return attributes_filter.apply_events(
        event_log,
        parameters={"attribute_key": "concept:name", "positive": False},
        values=activity_names,
    )


# ---------------------------------------------------------------------------
#  Activity helpers
# ---------------------------------------------------------------------------

def get_unique_activities(event_log):
    """
    Returns the set of all distinct activity names in a log.

    Args:
        event_log: PM4Py EventLog.

    Returns:
        set of str.
    """
    return {event["concept:name"] for trace in event_log for event in trace}


# ---------------------------------------------------------------------------
#  Fragment relabeling — renumber fragment_n_start/end to contiguous indices
# ---------------------------------------------------------------------------

def relabel_fragments_in_tree(process_tree):
    """
    Traverses a process tree in execution order and renumbers fragment nodes
    so that they have contiguous indices (0, 1, 2, ...).

    Args:
        process_tree: PM4Py ProcessTree with labels like ``fragment_7_start``.

    Returns:
        tuple: (ProcessTree, fragment_mapping) where fragment_mapping is
               ``{old_id: new_id}``.
    """
    def _get_nodes_preorder(node):
        nodes = [node]
        for child in node.children:
            nodes.extend(_get_nodes_preorder(child))
        return nodes

    all_nodes = _get_nodes_preorder(process_tree)

    fragment_start_nodes = [
        node for node in all_nodes
        if isinstance(node.label, str) and re.match(r"fragment_\d+_start", node.label)
    ]
    # Preserve execution order (left-to-right)
    fragment_start_nodes.sort(key=lambda node: all_nodes.index(node))

    fragment_mapping = {}
    for new_id, node in enumerate(fragment_start_nodes):
        match = re.match(r"fragment_(\d+)_start", node.label)
        old_id = match.group(1)
        fragment_mapping[old_id] = str(new_id)
        node.label = f"fragment_{new_id}_start"

    for node in all_nodes:
        if isinstance(node.label, str):
            match = re.match(r"fragment_(\d+)_end", node.label)
            if match:
                old_id = match.group(1)
                if old_id in fragment_mapping:
                    node.label = f"fragment_{fragment_mapping[old_id]}_end"

    return process_tree, fragment_mapping


def relabel_fragments_in_event_log(event_log, fragment_mapping, label_key="concept:name"):
    """
    Applies *fragment_mapping* to rename fragment labels in an event log.

    Args:
        event_log: PM4Py EventLog.
        fragment_mapping (dict): ``{old_id: new_id}``.
        label_key (str): Event attribute holding the activity label.

    Returns:
        PM4Py EventLog with updated labels (shallow copy).
    """
    new_log = deepcopy(event_log)

    for trace in new_log:
        for event in trace:
            label = event[label_key]
            match = re.match(r"fragment_(\d+)_(start|end)", label)
            if match:
                old_id, position = match.groups()
                if old_id in fragment_mapping:
                    event[label_key] = f"fragment_{fragment_mapping[old_id]}_{position}"

    return new_log


# ---------------------------------------------------------------------------
#  Root abstraction — build an abstracted log from fragment start/end events
# ---------------------------------------------------------------------------

def build_root_abstraction(fragment_properties_list):
    """
    Constructs a new EventLog where each fragment's first and last events
    per case are relabeled as ``fragment_N_start`` / ``fragment_N_end``.

    This is the key step in creating the **root model** — an abstraction of
    the fragment subprocesses that preserves their temporal boundaries.

    Args:
        fragment_properties_list: List of per-fragment dicts, each containing
            ``{group_name: {"start_events": [...], "end_events": [...]}}``.

    Returns:
        PM4Py EventLog with relabeled start/end events per fragment.
    """
    root_log = EventLog()
    trace_index = {}

    for fragment_entry in fragment_properties_list:
        for group_name, properties in fragment_entry.items():
            labeled_events = (
                [(e, "start") for e in properties.get("start_events", [])] +
                [(e, "end")   for e in properties.get("end_events", [])]
            )

            for entry, label in labeled_events:
                case_id = entry["case_id"]
                event = dict(entry["event"])
                event["concept:name"] = f"{group_name}_{label}"

                if case_id not in trace_index:
                    new_trace = Trace()
                    new_trace.attributes["concept:name"] = case_id
                    trace_index[case_id] = new_trace
                    root_log.append(new_trace)

                trace_index[case_id].append(event)

    return sorting.sort_timestamp_log(root_log)


def get_fragment_first_last_events(fragment_log, include_end=True):
    """
    Extracts the first and last event of each trace in *fragment_log*.

    Args:
        fragment_log: PM4Py EventLog (projected to a single fragment).
        include_end (bool): If False, only start events are returned.

    Returns:
        tuple: (start_events, end_events), each a list of
               ``{"case_id": ..., "event": ...}``.
    """
    start_events = [
        {"case_id": trace.attributes.get("concept:name"),
         "event":   min(trace, key=lambda e: e["time:timestamp"])}
        for trace in fragment_log if len(trace) > 0
    ]
    end_events = (
        [
            {"case_id": trace.attributes.get("concept:name"),
             "event":   max(trace, key=lambda e: e["time:timestamp"])}
            for trace in fragment_log if len(trace) > 0
        ]
        if include_end else []
    )
    return start_events, end_events


# ---------------------------------------------------------------------------
#  Core mining — projection + inductive miner
# ---------------------------------------------------------------------------

def mine_process_tree(event_log, activity_names=None, noise_threshold=0.0):
    """
    Core mining step: optionally **project** the log to *activity_names*,
    then run the **inductive miner** to obtain a process tree.

    This is the shared kernel used by :func:`mine_fragment_subprocess`,
    :func:`mine_and_visualize_model`, and the root-model pipeline.

    Args:
        event_log: PM4Py EventLog.
        activity_names (list of str, optional): If given, the log is first
            projected to these activities (fragment projection).
        noise_threshold (float): Noise threshold for the inductive miner.

    Returns:
        tuple: ``(process_tree, projected_log)``
    """
    if activity_names is not None:
        projected_log = project_log_to_activities(event_log, activity_names)
    else:
        projected_log = event_log

    process_tree = discover_process_tree_inductive(projected_log, noise_threshold=noise_threshold)
    return process_tree, projected_log


# ---------------------------------------------------------------------------
#  Visualization — BPMN + optional heuristics net (separate from mining)
# ---------------------------------------------------------------------------

def visualize_process_model(process_tree, event_log=None):
    """
    Converts a process tree to BPMN and displays it.
    If *event_log* is provided, also shows a heuristics-net view.

    Args:
        process_tree: PM4Py ProcessTree object.
        event_log (PM4Py EventLog, optional): Used for heuristics-net
            visualisation when given.
    """
    bpmn_model = convert_to_bpmn(process_tree)
    bpmn_gviz = bpmn_visualizer.apply(bpmn_model)
    bpmn_visualizer.view(bpmn_gviz)

    if event_log is not None:
        heu_net = discover_heuristics_net(
            event_log, dependency_threshold=0, and_threshold=0, loop_two_threshold=0,
        )
        heu_gviz = hn_visualizer.apply(heu_net)
        hn_visualizer.view(heu_gviz)


# ---------------------------------------------------------------------------
#  Convenience wrapper — mine + visualise in one call
# ---------------------------------------------------------------------------

def mine_and_visualize_model(
    event_log,
    activity_names=None,
    noise_threshold=0.0,
    return_process_tree=False,
    show_plots=True,
    relabel_fragments=False,
):
    """
    Convenience wrapper: calls :func:`mine_process_tree` then
    optionally :func:`visualize_process_model` and relabel fragments.

    Args:
        event_log: PM4Py EventLog.
        activity_names (list of str, optional): Projection step.
        noise_threshold (float): Noise threshold for inductive miner.
        return_process_tree (bool): If True, return the ProcessTree object.
        show_plots (bool): Whether to display BPMN / heuristics-net plots.
        relabel_fragments (bool): Whether to renumber fragment labels
            (for root-model trees).

    Returns:
        tuple — varies by *return_process_tree* / *relabel_fragments*:
            ``(process_tree, event_log, fragment_mapping?)`` or
            ``(event_log, fragment_mapping?)``.
    """
    process_tree, projected_log = mine_process_tree(
        event_log, activity_names=activity_names, noise_threshold=noise_threshold,
    )

    result = ()
    if relabel_fragments:
        process_tree, fragment_mapping = relabel_fragments_in_tree(process_tree)
        projected_log = relabel_fragments_in_event_log(projected_log, fragment_mapping)
        result = (fragment_mapping,)

    if show_plots:
        visualize_process_model(process_tree, event_log=projected_log)

    if return_process_tree:
        return (process_tree, projected_log, *result)
    return (projected_log, *result)


# ---------------------------------------------------------------------------
#  Fragment subprocess — project → mine → metrics → start/end events
# ---------------------------------------------------------------------------

def mine_fragment_subprocess(
    event_log,
    fragment_activities,
    noise_threshold=0.0,
    include_end_events=True,
    compute_metrics=True,
    show_plots=False,
):
    """
    Mines a single fragment subprocess from the full event log:

        1. **Project** L to *fragment_activities*  (L ↦ L↓_{A_f})
        2. **Mine** a process tree wth the inductive miner
        3. Optionally compute quality metrics
        4. Record first/last events per case for later root-model construction

    Args:
        event_log: Full PM4Py EventLog.
        fragment_activities (list of str): Activity labels belonging to this
            fragment.
        noise_threshold (float): Noise threshold for the inductive miner.
        include_end_events (bool): Whether to also extract end events.
        compute_metrics (bool): If True, compute fitness / precision / F1 / CFC.
        show_plots (bool): Visualise the mined model.

    Returns:
        dict::
            ``{"metrics": {...} or None, "process_tree": ..., "fragment_log": ...,
                "start_events": [...], "end_events": [...]}``
    """
    # --- 1. Projection + 2. Mine (shared core) ---
    process_tree, fragment_log = mine_process_tree(
        event_log, activity_names=fragment_activities, noise_threshold=noise_threshold,
    )

    # --- 3. Quality metrics ---
    metrics = calculate_metrics(process_tree, fragment_log) if compute_metrics else None

    # --- 4. First / last events for root abstraction ---
    start_events, end_events = get_fragment_first_last_events(fragment_log, include_end=include_end_events)

    # --- 5. Visualise (optional) ---
    if show_plots:
        print (fragment_activities)
        visualize_process_model(process_tree, event_log=fragment_log)

    return {
        "metrics":       metrics,
        "process_tree":  process_tree,
        "fragment_log":  fragment_log,
        "start_events":  start_events,
        "end_events":    end_events,
    }


# ---------------------------------------------------------------------------
#  Full pipeline — mine all fragments + build root abstraction + mine root
# ---------------------------------------------------------------------------

def mine_all_fragment_models_and_root(
    event_log,
    fragments,
    include_end_events=True,
    compute_metrics=True,
    compute_fragment_trees=True,
    subprocess_noise_threshold=0.0,
    root_noise_threshold=0.0,
    show_fragment_plots=False,
    show_root_plot=True,
):
    """
    Full pipeline:

        1. For each fragment, **project** the log and mine a subprocess model.
        2. Build the **root abstraction** from fragment start/end events.
        3. Mine the **root model** from the abstracted log.
        4. Optionally compute mean fragment quality + root quality metrics.

    This function is independent of XES export — you can call it on its own
    and then pass the results to :func:`export_xes_by_fragments`.

    Args:
        event_log: Full PM4Py EventLog.
        fragments (list of list of str): Each element is the activity list
            for one fragment.
        include_end_events (bool): Whether to use end events in the root log.
        compute_metrics (bool): Whether to compute fitness / precision / F1 / CFC.
        compute_fragment_trees (bool): Whether to return the fragment process trees.
        subprocess_noise_threshold (float): Noise threshold for fragment models.
        root_noise_threshold (float): Noise threshold for the root model.
        show_fragment_plots (bool): Visualise each fragment model.
        show_root_plot (bool): Visualise the root model.

    Returns:
        tuple: ``(root_log, root_metrics, mean_fragment_metrics, fragment_mapping, fragment_trees)``
            — returns ``(root_log, None, None, fragment_mapping, fragment_trees)`` when
            *compute_metrics* is False.
            fragment_trees`` is a list of tuples ``(fragment_index, process_tree_string)`` for each fragment.
            
    """
    fragment_properties_list = []
    fragment_trees = []

    # --- 1. Mine each fragment subprocess ---
    for idx, fragment_activities in enumerate(fragments):
        group_name = f'fragment_{idx}'

        result = mine_fragment_subprocess(
            event_log, fragment_activities,
            noise_threshold=subprocess_noise_threshold,
            include_end_events=include_end_events,
            compute_metrics=compute_metrics,
            show_plots=show_fragment_plots,
        )

        if compute_metrics and result["metrics"] is not None:
            print(f"  {group_name} metrics: {result['metrics']}")

        if compute_fragment_trees:
            fragment_trees.append((idx, result["process_tree"].to_string()))
            fragment_properties_list.append({
                group_name: {
                    "metrics":      result["metrics"],
                    "start_events": result["start_events"],
                    "end_events":   result["end_events"],
                }
            })

    # --- 2. Build root abstraction ---
    root_log = build_root_abstraction(fragment_properties_list)

    # Merge with original START / END events if present
    start_end_log = project_log_to_activities(event_log, ['START', 'END'])
    if len(start_end_log) != 0:
        root_log = merge_event_logs_by_trace_id(root_log, start_end_log)

    # --- 3. Mine root model ---
    root_activity_names = list(get_unique_activities(root_log))
    process_tree, root_log = mine_process_tree(
        root_log, activity_names=root_activity_names, noise_threshold=root_noise_threshold,
    )

    # Relabel fragments to contiguous indices
    process_tree, fragment_mapping = relabel_fragments_in_tree(process_tree)
    root_log = relabel_fragments_in_event_log(root_log, fragment_mapping)

    if show_root_plot:
        visualize_process_model(process_tree, event_log=root_log)

    if compute_metrics:
        root_metrics = calculate_metrics(process_tree, root_log)
        mean_fragment_metrics = calculate_mean_quality_measures(fragment_properties_list)
        print(f"  Root model metrics: {root_metrics}")
        print(f"  Mean fragment metrics: {mean_fragment_metrics}")
        return root_log, root_metrics, mean_fragment_metrics, fragment_mapping, fragment_trees

    return root_log, None, None, fragment_mapping, fragment_trees


# ---------------------------------------------------------------------------
#  XES export — write sub-logs and metrics to disk (does NOT mine)
# ---------------------------------------------------------------------------

def export_xes_by_fragments(
    event_log,
    fragments,
    export_path,
    filename,
    method_name,
    root_log=None,
    root_metrics=None,
    mean_fragment_metrics=None,
    include_root=True,
):
    """
    Exports fragment sub-logs + optional root log as XES files, and
    optionally saves PM4Py quality metrics to ``.pm4py.metrics.txt``.

    .. note::

        This function does **not** mine process models.  Call
        :func:`mine_all_fragment_models_and_root` first and pass the
        results here.  Example::

            root_log, root_metrics, mean_metrics, _ = \\
                mine_all_fragment_models_and_root(event_log, fragments, ...)
            export_xes_by_fragments(event_log, fragments, export_path, filename,
                                    method, root_log=root_log,
                                    root_metrics=root_metrics,
                                    mean_fragment_metrics=mean_metrics)

    Args:
        event_log: Full PM4Py EventLog.
        fragments (list of list of str): Fragment activity groupings.
        export_path (str): Base export directory.
        filename (str): Log filename (e.g. ``BPIC12.xes.gz``).
        method_name (str): Scorer / method identifier (e.g. ``"bigram"``).
        root_log (PM4Py EventLog, optional): Pre-mined root log.  Required
            for ``include_root=True``.
        root_metrics (dict, optional): Pre-computed root quality metrics.
        mean_fragment_metrics (dict, optional): Pre-computed mean fragment metrics.
        include_root (bool): Export the root-log XES file.
    """
    if include_root:
        if root_log is None:
            raise ValueError("root_log is required when include_root=True. "
                             "Call mine_all_fragment_models_and_root first.")
        xes_exporter.apply(root_log, f'{export_path}/xes/{filename}.root.{method_name}.xes.gz')

    for idx, fragment_activities in enumerate(fragments):
        fragment_log = project_log_to_activities(event_log, fragment_activities)
        print(f'\nfragment_{idx}: {fragment_activities}')
        xes_exporter.apply(fragment_log, f'{export_path}/xes/{filename}.{idx}.{method_name}.xes.gz')

    if root_metrics is not None and mean_fragment_metrics is not None:
        metrics_path = f'{export_path}/{filename}.pm4py.metrics.txt'
        with open(metrics_path, 'a') as f:
            f.write(f'{method_name};{root_metrics};{mean_fragment_metrics}\n')


# ---------------------------------------------------------------------------
#  Parsing utilities
# ---------------------------------------------------------------------------

def parse_metrics_file(metrics_path):
    """
    Parses a ``.pfm.metrics.txt`` file and returns fragment information
    per method.

    Format per line::

        method;scoring_method;score;[fragment_activities_list]

    Args:
        metrics_path (str): Path to the metrics file.

    Returns:
        dict: ``{method: {"score": float, "fragments": [[str, ...], ...]}}``
    """
    import ast
    results = {}
    with open(metrics_path, 'r') as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 3:
                continue
            method = parts[0]
            score = float(parts[2])
            fragments_str = ";".join(parts[3:])
            fragments = ast.literal_eval(fragments_str)
            results[method] = {"score": score, "fragments": fragments}
    return results


def get_xes_dir_and_base(metrics_path):
    """
    Derives the XES sub-log directory and the original log's filename base
    from a ``.pfm.metrics.txt`` path.

    Example::

        get_xes_dir_and_base("../data/processed/BPIC/BPIC12.xes.gz.pfm.metrics.txt")
        # => ("../data/processed/BPIC/xes", "BPIC12.xes.gz")

    Args:
        metrics_path (str): Path to a ``.pfm.metrics.txt`` file.

    Returns:
        tuple: (xes_directory, filename_base)
    """
    export_dir = os.path.dirname(metrics_path)
    metrics_filename = os.path.basename(metrics_path)
    base = metrics_filename[:-len(".pfm.metrics.txt")]
    xes_dir = os.path.join(export_dir, "xes")
    return xes_dir, base


def save_process_trees(process_trees, output_path):
    """
    Saves process-tree string representations to a text file.

    Format::

        method
          fragment_index: tree_string

    Args:
        process_trees (dict): ``{method: [(frag_idx, tree_string), ...]}``
        output_path (str): Path to the output ``.txt`` file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for method, trees in process_trees.items():
            f.write(f'{method}\n')
            for frag_idx, tree_str in trees:
                f.write(f'  {frag_idx}: {tree_str}\n')
            f.write('\n')


# # ---------------------------------------------------------------------------
# #  Domain-knowledge fragment helpers  (kept as reference)
# # ---------------------------------------------------------------------------

# def get_fragments_by_labels_from_log(event_log):
#     """
#     Groups activities by their first-letter or numeric prefix (excluding
#     ``START`` / ``END``).  Used as a simple domain-knowledge baseline.

#     Args:
#         event_log: PM4Py EventLog.

#     Returns:
#         list of list of str: Activity groupings.
#     """
#     activities = get_unique_activities(event_log)
#     return get_fragments_by_labels(activities)


# def get_fragments_by_labels(activities):
#     """
#     Groups activity names by label prefix (first character, or first two
#     underscore-separated parts if the name starts with digits).

#     Args:
#         activities (list of str): Activity names.

#     Returns:
#         list of list of str: Grouped activity names.
#     """
#     grouped = defaultdict(list)
#     activities = [a for a in activities if a not in ["START", "END"]]

#     for item in activities:
#         if item[:2].isdigit():
#             parts = item.split("_")
#             prefix = "_".join(parts[:2])
#         else:
#             prefix = item[0]
#         grouped[prefix].append(item)

#     return list(grouped.values())


# ---------------------------------------------------------------------------
#  Deprecated / unused — kept for reference but commented out
# ---------------------------------------------------------------------------

# def calculate_quality_measures(event_log):
#     """
#     [UNUSED]  Wrapper that mines a process tree from *event_log* and
#     computes quality metrics in one call.
#     """
#     process_tree = inductive_miner.apply(event_log)
#     return calculate_metrics(process_tree, event_log)


# def plot_pm4py_inductive_miner_bpmn(event_log, quality_measures=True):
#     """
#     [UNUSED]  Mines a process tree, displays BPMN + Petri net, and
#     optionally prints quality measures.
#     """
#     process_tree = discover_process_tree_inductive(event_log, noise_threshold=0.2)
#     net, im, fm = pt_converter.apply(process_tree)
#     bpmn_model = convert_to_bpmn(process_tree)
#     gviz = bpmn_visualizer.apply(bpmn_model)
#     bpmn_visualizer.view(gviz)
#     gviz = pn_visualizer.apply(net, im, fm)
#     pn_visualizer.view(gviz)
#     if quality_measures:
#         print(calculate_metrics(process_tree, event_log))
