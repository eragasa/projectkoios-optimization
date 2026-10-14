# Inspected references

## Inspection record

The cited book was inspected in the linked second-edition text rather than cited from
memory. The relevant pinpoint is Chapter 16, Section 16.4, book pages 464–466, especially
conditions (16.37) and Theorem 16.4: KKT conditions are sufficient for a convex quadratic
program and positive definiteness gives a unique global solution. Section 16.5, book
pages 467–470, states that a known active set reduces the problem to an equality-
constrained quadratic subproblem and describes active-set optimality detection.

The implementation follows the primal active-set structure described in Section 16.5:
solve the equality-constrained quadratic on the working face, take the largest feasible
step to a blocking constraint, and release an active constraint when its multiplier sign
is invalid. The repository adaptation specializes those steps to coordinate bounds,
starts from the clipped center, checks row-local binary64 KKT tolerances, releases the
smallest-index violating bound, activates simultaneous endpoint hits in coordinate order,
and imposes a positive iteration limit. No source code or pseudocode is copied.

## Academic references

- Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, second edition,
  Springer, 2006, Chapter 16, Sections 16.4–16.5, pp. 464–470.
  ISBN-13: 978-0-387-30303-1. Inspected text:
  <https://deeplearningpku.github.io/literature/Nocedal_Wright_Numerical_optimization_v2.pdf>.

## Software documentation

- NumPy, `numpy.linalg.solve`, consulted for the dense square-system API and its declared
  `LinAlgError` failure:
  <https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html>.
