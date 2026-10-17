# Mathematics and deterministic algorithm

## Gradient and optimality conditions

For

$$
q(x)=m+(x-c)^TQ(x-c),
$$

the full gradient is

$$
g(x)=\nabla q(x)=2Q(x-c).
$$

For each coordinate, the box KKT conditions are

$$
\begin{aligned}
\ell_i < x_i < u_i &\Longrightarrow g_i=0,\\
x_i=\ell_i<u_i &\Longrightarrow g_i\geq 0,\\
x_i=u_i>\ell_i &\Longrightarrow g_i\leq 0.
\end{aligned}
$$

When $\ell_i=u_i$, the coordinate is fixed and its gradient has no sign restriction.
For a convex quadratic these conditions are sufficient for a global solution. Strict
positive definiteness makes that solution unique.

The reported per-coordinate violation is

$$
v_i=\begin{cases}
|g_i|,&\text{free},\\
\max(0,-g_i),&\text{lower},\\
\max(0,g_i),&\text{upper},\\
0,&\text{fixed},
\end{cases}
$$

and the retained diagnostic is $v_{\max}=\max_i v_i$.

## Free-subspace step

At one active-set state, let $F$ contain the free coordinates and let every coordinate in
$A$ be fixed temporarily at its current lower, upper, or fixed endpoint. With
$d=x-c$, the exact Newton step on the current face solves

$$
Q_{FF}p_F=-(Qd)_F,\qquad p_A=0.
$$

Every principal submatrix $Q_{FF}$ of a positive-definite matrix is positive definite,
so this direction is unique. The backend never forms an inverse. It chooses the largest
$\alpha\in[0,1]$ for which $x+\alpha p$ remains in the box. Coordinates reached at the
blocking step become active. Exact endpoint values retain that endpoint. Strictly
interior values are compared with the lower endpoint first and then the upper endpoint;
therefore lower has deterministic precedence only when an interior value is accepted
against both endpoints.

## Active-set transitions and completeness

The initial feasible point is the unconstrained center clipped coordinatewise into the
box. Fixed coordinates remain fixed. A nonfixed coordinate at an endpoint begins active;
all other coordinates begin free.

Each iteration performs an independent row-local KKT check:

1. if any free coordinate is nonstationary, solve the free-subspace Newton system and
   take its largest feasible step;
2. otherwise, if a lower- or upper-active sign condition fails, release the smallest-index
   violating coordinate; or
3. otherwise, all box KKT conditions are accepted and the result is returned.

In exact arithmetic, this is the standard primal active-set structure for a strictly
convex quadratic: face minimization decreases the objective unless the current face is
stationary; a stationary face with an invalid multiplier releases a descent direction;
and a KKT point is the unique global minimizer. Binary64 execution additionally requires
the declared row-local tolerances and can fail rather than return a point.

The solver checks at most `maximum_iterations` active-set states. The default is 10,000.
Reaching the limit or stagnating before KKT acceptance raises `RuntimeError`. This gives a
finite execution boundary for every call without claiming that all admitted binary64
problems converge within the default limit.

## Ordering and ties

Coordinates are always inspected in ascending tuple order. Exact simultaneous bound hits
are activated in that order. When multiple active sign conditions fail at a stationary
face, the smallest coordinate index is released. Exact lower and upper hits retain their
endpoint; lower-endpoint tolerance snapping precedes upper snapping only for a strictly
interior value. These rules and the fixed initial point make repeated execution
deterministic for the same backend and binary64 inputs. Tolerance changes may change the
accepted active set and are therefore retained as part of the solver contract.

## Numerical acceptance

`ConstrainedQuadraticTolerance` controls two numerical checks:

1. a computed free coordinate may be snapped to an endpoint only when the coordinate and
   endpoint satisfy its scale-aware scalar comparison; and
2. every coordinate's final KKT violation must satisfy its own row-local bound.

For $d=x-c$, the row-local arithmetic scale and acceptance bound are

$$
s_i=2\sum_j |Q_{ij}|\,|d_j|,
\qquad
 t_i=a+r\max(1,s_i),
$$

where $a$ and $r$ are the declared absolute and relative tolerances. Acceptance requires
$v_i\leq t_i$ independently for every coordinate. A large correctly signed gradient on
one active coordinate therefore cannot mask nonstationarity on a different free
coordinate. The summation is checked for overflow and nonfinite terms.

The result retains every $t_i$, $v_{\max}=\max_i v_i$, and
$\rho_{\max}=\max_i(v_i/t_i)$. Its sole frozen constructor accepts and validates
canonical exact binary64 tuples and scalars. Because at least one tolerance component is
positive, every $t_i$ is positive. The accepted result has $\rho_{\max}\leq 1$. These values are
software acceptance diagnostics, not a backward-error bound, exact certificate, or
consumer scientific resolution.

## Arithmetic failures

Nonfinite free-subspace solves, point construction, gradient products, gradient sums, loss
evaluation, or tolerance bounds fail with `FloatingPointError`. Existing
`PositiveDefiniteQuadraticLoss.squared_loss()` implementations classify nonfinite
internal evaluation with `ValueError`; after candidate shape, finiteness, and feasibility
have already been validated, the box capability narrowly translates that arithmetic
`ValueError` to a chained `FloatingPointError`. A `numpy.linalg.LinAlgError` from a
principal solve, pre-optimality stagnation, or exhaustion of `maximum_iterations` is
reported as `RuntimeError`; only the linear-algebra error is chained from a backend
exception. Because the accepted input matrix is positive definite, these are backend
numerical failures rather than alternative optimization results.
