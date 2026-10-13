# Pareto-shell contrastive sampling

- **Recorded:** 2026-10-13T03:54:09+08:00
- **Originator:** Eugene Joseph M. Ragasa
- **Status:** owner-authored research method; not historical PyPosPack reconstruction
- **Disclosure status:** local, uncommitted design record

## Idea

A one-pass Pareto filter discards useful directional evidence. Instead, recursively peel
the evaluated objective vectors into zero-based nondominated shells:

\[
P_0 = \operatorname{Pareto}(Y),
\]

\[
P_k = \operatorname{Pareto}\left(Y \setminus \bigcup_{j=0}^{k-1}P_j\right).
\]

Pull the indices in every shell back to their exact decision vectors. Shell zero is
favorable relative to the observed population. Shells one and two are progressively
less favorable and provide negative spatial evidence rather than being discarded.

The proposed sampling update fits a decision-space density \(q_k(x)\) to each shell. A
bounded candidate pool drawn from \(q_0\) is ranked by a contrastive score such as

\[
s(x) = \log q_0(x)
       - \log\left(\lambda_1q_1(x)+\lambda_2q_2(x)\right).
\]

Selecting high-scoring candidates keeps proposals near the observed shell-zero geometry
while favoring the side with less density from dominated shells. Repeating external
forward evaluation, shell construction, pullback, and contrastive sampling supplies
implicit distributional selection pressure without differentiating or globally
approximating the forward evaluator.

## Ownership and historical boundary

Repository and Git-history searches found the historical PyPosPack implementation of
one-pass Pareto membership, Pareto pullback, global KDE, and clustered KDE. They did not
find recursive Pareto-shell pullback or shell-conditioned positive/negative KDE
sampling. This method must therefore remain visibly separate from the provenance-bound
PyPosPack Pareto kernel and the historical GMM/DBSCAN reconstruction.

The implementation may call the preserved Pareto selector, but it must not be described
as recovered donor behavior. Equal objective vectors are grouped before shell peeling so
they receive the same shell rank despite the donor kernel's historical weak-dominance
duplicate semantics.

## Closest known concepts, not equivalence claims

Nondominated sorting and Pareto shells are established concepts. Related methods include
rank-based multi-objective evolutionary selection, 2017 HypI shell scalarization,
Parzen/TPE good-versus-bad density ratios, 2020--2022 MOTPE/MBORE, and negative updates
in active covariance adaptation. No novelty or patentability conclusion is made here.
The specific combination of recursive shell pullback and contrastive decision-space KDE
requires a dedicated literature review before publication claims.

## Verification questions

At equal external-evaluation budgets, compare:

1. bounded uniform sampling;
2. shell-zero global KDE;
3. historical GMM- or DBSCAN-conditioned KDE;
4. two-shell contrastive KDE;
5. three-shell contrastive KDE.

Measure new nondominated points, hypervolume improvement where scientifically
appropriate, archive coverage, invalid proposals, duplicate evaluations, and evaluation
count to a declared target. Separately test sensitivity to shell weights, proposal-pool
size, bandwidth, covariance stabilization, archive size, decision dimension, and
objective dimension.

## First forward-evaluation benchmark

The initial deterministic benchmark is recorded in
`benchmarks/results/pareto_shells_forward_2000.json`. Every proposal was passed through a ZDT
or DTLZ2 forward evaluator; reference fronts were used only for post-budget IGD scoring.
At 2,000 evaluations and three seeds, the contrastive sampler improved over uniform on
10-D ZDT1 and ZDT3, but did not consistently beat global KDE and was worse than uniform
on 30-D DTLZ2. Both KDE approaches failed on canonical 30-D ZDT1 and ZDT3 because the
shell-zero pullback contained fewer samples than decision dimensions, exposing the
expected singular-covariance blocker.

Typical pymoo 0.6.2 NSGA-II used population and offspring batches of 100 and twenty
sequential feedback batches. It achieved the lowest IGD on every successful smooth
benchmark. A second NSGA-II configuration used the same initial 1,000 decision vectors,
500 offspring, and three batches. Under that batch-matched schedule, shell contrast beat
NSGA-II on both 10-D ZDT problems and global KDE beat it on both DTLZ2 configurations.
This separates evaluation budget from the number of selection/adaptation barriers.

A second run increased the budget to 20,000 evaluations: `10,000 + 5,000 + 5,000` for
the three-batch strategies and 200 populations for typical NSGA-II. Typical NSGA-II won
decisively on every completed smooth benchmark. Global KDE was the best fully replicated
three-batch method. Shell contrast beat uniform on both 10-D ZDT problems and 12-D
DTLZ2, but lost to global KDE. On canonical 30-D ZDT, only one of three KDE seeds formed
a full-rank shell even after 10,000 initial observations.

A controlled 20-dimensional irregular suite then added a curved coupled manifold, two
disconnected pullback modes, rugged modulation, discontinuous mode switching with
invalid evaluations, deterministic bounded observation pseudo-noise, and a
three-objective multimodal case with enough Pareto observations for a genuine
multi-component GMM. At 20,000 evaluations, stabilized GMM-conditioned KDE achieved the
best three-batch IGD on the rugged and noisy cases; global KDE won two cases; shell
contrast narrowly won the three-objective case. Typical 200-batch NSGA-II remained best
in IGD.

Decision-space mode scoring exposed a different result: GMM-KDE retained both equivalent
pullback modes in every successful multimodal run, while typical NSGA-II collapsed to
one mode on both two-objective cases and nearly collapsed on the three-objective case.
The reconstructed historical exact-zero covariance repair remained insufficient for
floating-point rank loss. The separately identified relative eigenspectrum floor
completed every stabilized GMM seed, including canonical 30-D ZDT. On the failure-region
problem, all KDE methods still had lower valid-evaluation yield than uniform, while
NSGA-II learned the feasible tube.

These results reject any claim that the first global density-ratio rule is already a
finished high-dimensional method. Negative Pareto shells remain potentially useful, but
the directional update, feasibility treatment, and covariance treatment require
controlled alternatives and larger replication.

## Known risks

- Exact dominance becomes less selective as objective dimension grows.
- KDE density ratios can be unstable in sparse high-dimensional decision spaces.
- Hard repulsion can eliminate valid disconnected modes represented in lower shells.
- Lower shell rank is only population-relative evidence, not proof that a decision-space
  region is globally unfavorable.
- A symmetric shell-zero KDE plus contrastive ranking creates only a bounded stochastic
  extrapolation; it is not a mathematical gradient of the forward evaluator.
- Exploration mass or a uniform mixture may be needed to prevent premature collapse.
