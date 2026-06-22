# ProcessFragmentMiner (PFM)

**Hierarchical process mining** — automatically discovers subprocess *fragments* from an event log and composes them into a two-level (root + fragment) process model.

## Author

Joern Tobis — joern.tobis@tum.de  
License: AGPL-3.0 (commercial use requires prior written permission).

---

## Overview

PFM takes a PM4Py event log (XES) and produces:

1. A set of **disjoint fragments** — each fragment is a group of activities that form a coherent subprocess.
2. A **subprocess model** (process tree + Petri net) for each fragment, mined via the inductive miner on a projected log.
3. A **root abstraction** — the original log is abstracted by replacing each fragment's first and last events with `fragment_N_start` / `fragment_N_end` markers. A root process tree is mined from this abstracted log, capturing the high-level orchestration of fragments.

---

## Pipeline

```
Event Log
    │
    ▼
Dependency Graph (Heuristics Miner) ────► Subtrace Extraction (DFS)
    │                                               │
    │                                               ▼
    │                                    Candidate subtraces (scored)
    │                                               │
    │                                               ▼
    │                                    Disjoint subset selection (DP / beam)
    │                                               │
    ▼                                               ▼
Fragment Subprocesses ──────────────────► Fragment definitions
    │                                               │
    │       ┌─── Project log to fragment activities │
    │       │   └─► Mine subprocess tree            │
    │       │   └─► Compute quality metrics         │
    │       │   └─► Record start/end events         │
    │       ▼                                       │
    │   Fragment models                              │
    │       │                                       │
    │       ▼                                       │
    │   Build Root Abstraction ◄────────────────────┘
    │       │   (fragment_N_start / fragment_N_end)
    │       ▼
    │   Mine Root Model
    ▼
Two-level hierarchy: Root + Fragment trees
```

---

## Scoring strategies

| Scorer | File | Description |
|---|---|---|
| **BigramScorer** | `scorer.py:19` | Laplace-smoothed bigram probability — how likely is an activity sequence given training data. |
| **DependencyScorer** | `scorer.py:70` | Product of Heuristics Miner dependency strengths along the path. |
| **SimilarityScorer** | `scorer.py:89` | Average Word2Vec (gensim, skip-gram) cosine similarity between consecutive activities. |

Custom scorers can be registered via `ScorerFactory.register()` (`scorer_factory.py`).

---

## Module structure

| File | Purpose |
|---|---|
| `miner.py` | `ProcessFragmentMiner` — the main orchestrator class. |
| `subtrace_extractor.py` | DFS-based enumeration of high-scoring subtraces from the dependency graph. Uses a heap to keep the top-*k* per start node. |
| `fragment_selector.py` | Selects the best **disjoint** subset of subtraces. Implements exact DP (bitmask-based, exponential) with automatic fallback to beam search when memory exceeds `max_memory_mb`. |
| `scorer.py` | `BaseScorer`, `BigramScorer`, `DependencyScorer`, `SimilarityScorer`. |
| `scorer_factory.py` | `ScorerFactory` — creates scorer instances from string names, resolving parameters from the miner automatically. |
| `utils.py` | All PM4Py wrappers: log projection, inductive mining, quality metrics (fitness, precision, F1, CFC), root abstraction construction, fragment relabeling, and XES export. |
| `test.py` | `evaluation()` — end-to-end evaluation pipeline. Loads logs, runs all scorers, exports fragments and metrics. |
| `adapters/pm4py_adapter.py` | Loads XES logs and extracts dependency graphs via the Heuristics Miner. |
| `adapters/word2vec_adapter.py` | Trains a gensim Word2Vec model on traces; exposes `similarity()` and `contains()`. |

---

## Fragment selection algorithm

The disjoint-subset problem is NP-hard (weighted set packing). PFM offers two solvers:

1. **DP** (`_dp_solver`): exact, tracks state by a bitmask of covered events. Memory grows as O(2^n). Falls back automatically when usage exceeds `max_memory_mb`.
2. **Beam search** (`_beam_solver`): approximate heuristic that keeps the `beam_size` best partial solutions at each step.

Both support three aggregation functions: `sum`, `mean`, `log_likelihood`, plus an optional coverage bonus (`alpha`).

---

## Coverage guarantee

After selection, `ProcessFragmentMiner._ensure_full_coverage()` (`miner.py:129`) appends any uncovered activity as a singleton fragment, guaranteeing every event name appears in at least one fragment.

---

## Dependencies

- Python ≥ 3.11
- pm4py ≥ 2.7.16
- gensim ≥ 4.3.3
- scikit-learn ≥ 1.7.0
- prettyprinttree ≥ 2.0.1
- Managed via [uv](https://github.com/astral-sh/uv) (`uv sync`)

---

## Usage

```python
from process_fragment_miner import ProcessFragmentMiner, evaluation

# Basic usage
miner = ProcessFragmentMiner(event_log, scorer="bigram")
subtraces = miner.extract_subtraces(max_depth=5, min_depth=2, top_k=10)
score, fragments = miner.mine_best_fragments(subtraces)

# Full evaluation pipeline
evaluation(logs_dir="data/", export_path="results/", methods=("bigram", "dependency", "similarity"))
```

---

## Citation

```bibtex
@software{tobis_processfragmentminer,
  author = {Joern Tobis},
  title  = {ProcessFragmentMiner},
  year   = {2025},
  doi    = {...}
}
```

Or use the included `CITATION.cff`.
