# Python test policy

## Evidence

- Each test identifies the public invariant, mathematical identity, boundary, or
  malformed backend contract under verification.
- Expected values are derived independently from the production operation under test.
- Production output is never copied into a test oracle.
- Expected values are never changed merely to make a failing test pass.
- Every oracle or fixture correction records the mathematical derivation or contract
  correction that invalidates the previous expectation.

## Record validation

Record tests distinguish rejected public operations from mechanisms that bypass the
record contract and fabricate unreachable state.

- Invalid-state tests use explicit canonical constructor inputs.
- Each invalid input corresponds to a named contract condition.
- Tests may assert that ordinary field assignment raises `FrozenInstanceError` or
  `AttributeError`; the rejected assignment leaves the record unchanged.
- Invalid-state tests do not clone a valid production record and arbitrarily perturb its
  fields through `object.__setattr__`, `object.__new__`, `dataclasses.replace`, generic
  override dictionaries, or equivalent mechanisms.
- Backend trust-boundary tests use typed immutable probes when a public constructor
  cannot represent the malformed state.

## Public boundaries

- Tests import only public production symbols.
- Tests do not import underscore-prefixed production symbols.
- Constructor invariants are verified through public constructors.
- Solver behavior is verified through public solver methods.
- Operations requiring direct independent verification are promoted to documented public
  capabilities rather than treated as private-helper test contracts.
- Public nominal extension points use public handoff records.
- Extension points that require private handoff records remain closed.
