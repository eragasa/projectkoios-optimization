# Forward Pareto benchmarks

`pareto_shells_forward.py` evaluates every proposed decision vector through an explicit
benchmark forward evaluator. Samplers and optimizers receive only prior decision/objective
observations; they do not receive benchmark equations, gradients, inverse mappings, or
reference-front points. Known fronts are used only after the evaluation budget is
exhausted to calculate inverted generational distance (IGD; lower is better).

The compared strategies are:

- independent bounded uniform sampling;
- global Gaussian KDE fitted to Pareto shell zero;
- PyPosPack-lineage GMM-conditioned local KDE with both the historical exact-zero
  covariance repair and a modern relative eigenspectrum floor;
- Pareto-shell contrastive KDE using shell-zero proposal density and shell-one/two
  negative densities;
- typical pymoo 0.6.2 NSGA-II with population/offspring size 100; and
- batch-matched pymoo NSGA-II using the declared initial/adaptive stage sizes, the same
  initial SplitMix64 population, and the same three adaptation barriers as the Koios
  strategies.

The explicit NSGA-II configuration uses bounded random initialization, binary tournament
selection by rank and crowding, SBX crossover with probability `0.9` and `eta=15`,
polynomial mutation with individual probability `0.9`, variable probability `1/n`, and
`eta=20`, rank-and-crowding survival, and duplicate elimination. No problem-specific
parameters are tuned.

All Koios strategies use the same first uniform stage for a given problem and seed.
Gaussian KDE proposals are deterministically reflected into the benchmark's `[0, 1]`
decision box; reflection is benchmark boundary handling, not part of the maintained
samplers.

Run the recorded configuration with the `benchmark` extra installed:

```bash
PYTHONPATH=src/python .venv/bin/python benchmarks/pareto_shells_forward.py \
  --seeds 3 --initial 1000 --adaptive 500 --rounds 2 \
  --output benchmarks/results/pareto_shells_forward_2000.json
```

This spends exactly 2,000 forward evaluations per successful run. The recorded output is
`results/pareto_shells_forward_2000.json`.

## Initial 2,000-evaluation result

Median cumulative-archive IGD over three deterministic seeds:

| Problem | Uniform | Global KDE | Stabilized GMM-KDE | Shell contrast | NSGA-II, 3 batches | Typical NSGA-II, 20 batches |
|---|---:|---:|---:|---:|---:|---:|
| ZDT1, 10 decisions | 1.2874 | 0.8751 | 0.7244 | **0.6277** | 0.9675 | **0.1788** |
| ZDT3, 10 decisions | 1.0818 | 0.7222 | **0.6465** | 0.6601 | 1.0768 | **0.1514** |
| DTLZ2, 12 decisions, 3 objectives | 0.3054 | **0.2265** | 0.2351 | 0.3073 | 0.3062 | **0.1004** |
| DTLZ2, 30 decisions, 3 objectives | 1.2525 | 1.1318 | **1.0827** | 1.3122 | 1.2812 | **0.5540** |

Bold within the first five columns marks the best three-batch strategy. The final column
is the best result overall on every successful smooth benchmark, but it obtains twenty
selection/reproduction feedback cycles rather than three. The batch-matched NSGA-II
comparison isolates that distinction. At this small budget, shell contrast wins the
three-batch ZDT1 case, stabilized GMM-KDE wins ZDT3 and 30-D DTLZ2, and global KDE wins
12-D DTLZ2. NSGA-II loses most of its advantage when restricted to the same three
adaptive batches.

These three-seed results are exploratory software/numerical evidence, not scientific
validation. Publication evidence needs more seeds, confidence intervals, hypervolume or
IGD+ as appropriate, and an explicit parallel-resource model. Adaptation barriers alone
do not determine wall time: with limited workers, a large stage requires several
forward-evaluation waves.

## 20,000-evaluation result

The larger run used `10,000 + 5,000 + 5,000` proposals for the three-batch strategies
and 200 populations of 100 for typical NSGA-II:

```bash
PYTHONPATH=src/python .venv/bin/python benchmarks/pareto_shells_forward.py \
  --seeds 3 --initial 10000 --adaptive 5000 --rounds 2 \
  --output benchmarks/results/pareto_shells_forward_20000.json
```

Median cumulative-archive IGD:

| Problem | Uniform | Global KDE | Stabilized GMM-KDE | Shell contrast | NSGA-II, 3 batches | Typical NSGA-II, 200 batches |
|---|---:|---:|---:|---:|---:|---:|
| ZDT1, 10 decisions | 0.8754 | **0.3703** | 0.4845 | 0.4247 | 0.8081 | **0.000269** |
| ZDT3, 10 decisions | 0.8800 | 0.3109 | **0.3033** | 0.3381 | 0.6284 | **0.000382** |
| DTLZ2, 12 decisions, 3 objectives | 0.2111 | **0.1166** | 0.1196 | 0.1876 | 0.2114 | **0.01227** |
| DTLZ2, 30 decisions, 3 objectives | 1.1010 | **0.8667** | 0.8689 | 1.1138 | 1.0398 | **0.02310** |

Typical NSGA-II wins decisively after 200 feedback cycles on these smooth problems.
Among methods restricted to three batches, global KDE wins three cases and stabilized
GMM-KDE narrowly wins ZDT3. The GMM often selects one component when a two-objective
Pareto archive is small, so these problems do not strongly exercise mixture behavior.

The canonical 30-D ZDT global, shell, and historical-repair GMM results are omitted
because only one of three seeds completed at 20,000 evaluations. Stabilized GMM-KDE
completed all seeds: IGD `1.3835` on ZDT1 and `1.2521` on ZDT3. Typical NSGA-II remained
far better (`0.00214` and `0.00125`).

Figures:

- [`pareto_shells_forward_20000_igd.png`](results/pareto_shells_forward_20000_igd.png) — all strategies on a logarithmic IGD scale;
- [`pareto_shells_forward_20000_igd_three_batch.png`](results/pareto_shells_forward_20000_igd_three_batch.png) — detailed comparison at three adaptation batches.

Regenerate them with:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python \
  benchmarks/plot_forward_results.py \
  benchmarks/results/pareto_shells_forward_20000.json \
  benchmarks/results/pareto_shells_forward_20000_igd
```

## Irregular 20-dimensional suite

A second suite keeps known analytical reference fronts while making their decision-space
pullbacks difficult. Five problems use a ZDT1 front with one front coordinate and
nineteen coupled nuisance coordinates; a sixth uses a three-objective DTLZ2 sphere with
two front coordinates and eighteen coupled nuisance coordinates:

- **coupled manifold** applies a fixed rotation, unequal scales, and adjacent-coordinate
  coupling around a curved one-dimensional decision manifold;
- **multimodal pullback** gives the same objective front two disconnected decision-space
  manifolds;
- **rugged multimodal** adds high-frequency deterministic modulation away from both
  manifolds;
- **switched failures** alternates discontinuously between manifolds and rejects
  decisions outside a feasible tube; and
- **noisy coupled** adds bounded deterministic pseudo-noise to the observed objective
  vectors while IGD is calculated from noise-free objectives at the decisions selected
  by each optimizer; and
- **multimodal three-objective** supplies a large Pareto seed population with two
  disconnected pullback manifolds, allowing BIC to select a genuine multi-component GMM.

The forward evaluator exposes only observed objective vectors and finite failure
penalties. Clean objectives and reference fronts are scoring-only data.

Run the suite with:

```bash
PYTHONPATH=src/python .venv/bin/python benchmarks/pareto_shells_forward.py \
  --suite irregular --seeds 3 --initial 10000 --adaptive 5000 --rounds 2 \
  --output benchmarks/results/pareto_shells_irregular_forward_20000.json
```

Median clean-objective IGD after 20,000 forward evaluations:

| Problem | Uniform | Global KDE | Stabilized GMM-KDE | Shell contrast | NSGA-II, 3 batches | Typical NSGA-II, 200 batches |
|---|---:|---:|---:|---:|---:|---:|
| Coupled manifold | 0.1995 | **0.0891†** | 0.0938 | 0.1166† | 0.1868 | **0.0152** |
| Multimodal pullback | 0.1590 | **0.0713** | 0.0729 | 0.0893 | 0.1626 | **0.0121** |
| Rugged multimodal | 0.1804 | 0.1002 | **0.0920** | 0.1053 | 0.1799 | **0.0268** |
| Switched failures | 0.1813 | **0.0857** | 0.0902 | 0.1033 | 0.1817 | **0.0275** |
| Noisy coupled | 0.1995 | 0.0918 | **0.0861** | 0.1344 | 0.1819 | **0.0281** |
| Multimodal, 3 objectives | 0.0593 | 0.0451 | 0.0439 | **0.0431** | 0.0589 | **0.0198** |

† Only two of three global/shell coupled-manifold seeds completed because one seed still
had a rank-deficient shell. Stabilized GMM-KDE completed all three. Historical-repair
GMM-KDE also completed only two coupled and two rugged seeds; where both repairs
completed on full-rank data, their results were numerically equal.

Typical NSGA-II remains best in objective-space IGD with 200 feedback cycles, but its
advantage is substantially smaller than on smooth ZDT. Under three adaptation barriers,
global KDE wins two fully replicated cases, stabilized GMM-KDE wins rugged and noisy
cases, and shell contrast narrowly wins the genuine multi-component three-objective
case. All three adaptive KDE strategies beat uniform and batch-matched NSGA-II on that
case.

On switched failures, median valid-evaluation fractions were `0.8321` for uniform,
`0.8049` for global KDE, `0.8044` for stabilized GMM-KDE, `0.7880` for shell contrast,
`0.8776` for batch-matched NSGA-II, and `0.9725` for typical NSGA-II. The current KDE
samplers therefore do not yet learn feasibility effectively.

IGD alone hides decision-space mode collapse because both disconnected parameter
families map to the same objective front. Median archive mode balance uses zero for one
surviving mode and one for equal representation:

| Problem | Global KDE | Stabilized GMM-KDE | Shell contrast | Typical NSGA-II |
|---|---:|---:|---:|---:|
| Multimodal pullback | 0.744 | **0.767** | 0.656 | 0.000 (one mode) |
| Rugged multimodal | 0.325 | 0.424 | **0.793** | 0.000 (one mode) |
| Multimodal, 3 objectives | **0.904** | 0.780 | 0.806 | 0.003 |

Thus GMM-KDE retained both exact decision-space modes in every successful multimodal
run, while typical NSGA-II collapsed to one mode on both two-objective cases and nearly
collapsed on the three-objective case despite its superior IGD. The recorded median GMM
component history for the three-objective case was `(4, 4)` across its two adaptive
stages, so it exercises actual mixture conditioning rather than a one-cluster fallback.

Figures:

- [`pareto_shells_irregular_forward_20000_igd.png`](results/pareto_shells_irregular_forward_20000_igd.png);
- [`pareto_shells_irregular_forward_20000_igd_three_batch.png`](results/pareto_shells_irregular_forward_20000_igd_three_batch.png);
- [`pareto_shells_irregular_forward_20000_igd_valid_fraction.png`](results/pareto_shells_irregular_forward_20000_igd_valid_fraction.png);
- [`pareto_shells_irregular_forward_20000_igd_mode_balance.png`](results/pareto_shells_irregular_forward_20000_igd_mode_balance.png).

These are controlled synthetic stress tests, not substitutes for real simulator cases.
The pseudo-noise is deterministic to make exact reruns possible; replicated stochastic
noise remains future work.

## Covariance reconstruction result

Canonical 30-decision ZDT1 and ZDT3 expose locally low-rank Pareto pullbacks. Global KDE,
shell contrast, and GMM-KDE with the reconstructed historical repair still fail because
the donor rule replaces only eigenvalues exactly equal to zero. Floating-point rank loss
also produces small negative eigenvalues, so the historical rule is insufficient.

The separately identified symmetric relative eigenspectrum floor completed every
stabilized GMM-KDE seed at both budgets. At 20,000 evaluations it achieved IGD `1.3835`
on ZDT1 and `1.2521` on ZDT3, improving over uniform and three-batch NSGA-II but remaining
far behind typical 200-batch NSGA-II. This establishes numerical completion, not
scientific validation. A reduced-coordinate model remains the more geometrically
motivated next comparison.
