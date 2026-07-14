# Scoring Methods

ProcessFragmentMiner uses scoring methods to evaluate and rank candidate fragment traces. Each scorer assigns a numeric score to a fragment (activity sequence), which is then used by the fragment selection algorithm to find the best disjoint set.

## Available Scorers

### 1. Frequency (`frequency`)

**What it measures:** How widely a fragment's activities appear across the event log.

**Formula:**
```
score(fragment) = (number of traces containing fragment's activities) / (total traces in log)
```

**Range:** [0, 1]

**Interpretation:**
- Score = 1.0 → fragment appears in every trace
- Score = 0.5 → fragment appears in half the traces
- Score = 0.0 → fragment appears in no traces

**Example:**
```
Log traces:
  [A, B, C, D]
  [A, B, E, F]
  [A, G, H, I]
  [J, K, L, M]

Fragment = [A, B]
→ Traces containing A and B: traces 1 and 2
→ Score = 2/4 = 0.5
```

**When to use:** When you want fragments that are common across cases, capturing the "core" of the process.

---

### 2. Heuristic / Dependency (`heuristic`)

**What it measures:** The strength of sequential dependencies between consecutive activities in the fragment, derived from the Heuristics Miner's dependency matrix.

**Formula:**
```
score(fragment) = ∏ dependency(activity_i, activity_{i+1})  for all consecutive pairs
```

**Range:** [0, 1] per edge; product across edges (smaller for longer fragments)

**Interpretation:**
- High score → activities strongly follow each other in the log
- Low score → activities rarely occur in sequence
- -inf → any consecutive pair has no observed dependency

**Example:**
```
Dependency matrix:
  A→B: 0.9
  B→C: 0.8
  C→D: 0.7

Fragment = [A, B, C]
→ Score = 0.9 × 0.8 = 0.72
```

**When to use:** When you want fragments that represent strong sequential patterns (Activity B reliably follows Activity A).

---

### 3. Bigram (`bigram`)

**What it measures:** The probability of the fragment's activity sequence under a smoothed bigram language model trained on the log.

**Formula:**
```
P(fragment) = P(a_1) × ∏ P(a_i | a_{i-1})  for all activities

With Laplace smoothing:
P(a_i | a_{i-1}) = (bigram_count(a_{i-1}, a_i) + α) / (unigram_count(a_{i-1}) + α × |V|)
```

**Range:** (0, 1] (always positive due to smoothing)

**Interpretation:**
- Higher score → sequence is more likely given the training data
- Lower score → sequence is less common or unusual
- Sensitive to fragment length (longer fragments → smaller scores due to multiplicative nature)

**Example:**
```
Training traces:
  [A, B, C]
  [A, B, D]
  [A, C, D]

Unigram counts: A=3, B=2, C=3, D=2
Bigram counts: (A,B)=2, (A,C)=1, (B,C)=1, (B,D)=1, (C,D)=2

Fragment = [A, B]
→ P(A) = (3 + 1) / (10 + 1×4) = 4/14 ≈ 0.286
→ P(B|A) = (2 + 1) / (3 + 1×4) = 3/7 ≈ 0.429
→ Score = 0.286 × 0.429 ≈ 0.123
```

**When to use:** When you want fragments that are statistically likely given the overall process behavior.

---

### 4. Similarity (`similarity`)

**What it measures:** The semantic similarity between consecutive activities in the fragment, using Word2Vec embeddings trained on the log.

**Formula:**
```
score(fragment) = mean(cosine_similarity(a_i, a_{i+1}))  for all consecutive pairs
```

**Range:** [-1, 1] (typically [0, 1] for process activities)

**Interpretation:**
- High score → consecutive activities are semantically similar
- Low score → consecutive activities are semantically different
- Requires Word2Vec embeddings (trained on the log)

**Example:**
```
Activity embeddings (simplified 2D):
  A = [0.9, 0.1]
  B = [0.8, 0.2]
  C = [0.1, 0.9]

Fragment = [A, B]
→ cosine(A, B) ≈ 0.98 (very similar)
→ Score = 0.98

Fragment = [A, C]
→ cosine(A, C) ≈ 0.28 (dissimilar)
→ Score = 0.28
```

**When to use:** When you want fragments with semantically coherent activity sequences.

---

### 5. Weighted (`weighted`)

**What it is:** A meta-scorer that combines multiple sub-scorers via weighted min-max normalization.

**How it works:**
1. Scores every trace with each sub-scorer to determine min/max bounds
2. Normalizes each sub-scorer's output to [0, 1] using min-max scaling
3. Computes weighted sum of normalized scores

**Formula:**
```
score(fragment) = Σ weight_k × normalize(sub_score_k(fragment))

where normalize(x) = (x - min) / (max - min)
```

**Range:** [0, 1] (weighted average of normalized sub-scores)

**Example:**
```python
ProcessFragmentMiner(
    event_log=log,
    scorer="weighted",
    scorer_kwargs={
        "scorers": [("heuristic", 0.5), ("frequency", 0.5)]
    },
)
```

**When to use:** When you want to balance multiple criteria (e.g., strong dependencies + high frequency).

---

## Summary Table

| Scorer | Range | What it captures | Best for |
|--------|-------|------------------|----------|
| `frequency` | [0, 1] | How common the fragment is | Core process activities |
| `heuristic` | [0, 1] | Sequential dependency strength | Strong causal patterns |
| `bigram` | (0, 1] | Statistical likelihood | Probable sequences |
| `similarity` | [-1, 1] | Semantic coherence | Semantically related activities |
| `weighted` | [0, 1] | Custom combination | Balanced multi-criteria |

---

## Choosing a Scorer

- **Exploratory analysis:** Start with `frequency` to find common patterns
- **Process compliance:** Use `heuristic` to find strong sequential rules
- **Anomaly detection:** Use `bigram` to find unlikely sequences
- **Semantic analysis:** Use `similarity` for activity grouping
- **Production use:** Use `weighted` to balance multiple objectives
