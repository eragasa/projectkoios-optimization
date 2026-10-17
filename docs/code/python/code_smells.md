# Python code-smell policy

## Typing

- Production annotations use capability-specific types.
- Production annotations do not use `Any`.
- Production parameter and return annotations do not use `object` when a narrower type
  expresses the contract.
- Production return annotations do not expose unresolved type parameters.
- Runtime validation uses explicit field access rather than reflective `getattr` or
  `__getattr__` dispatch.

## Immutable records

- Frozen records contain canonical storage when construction completes.
- Frozen records do not normalize fields through `object.__setattr__`.
- Public constructors reject noncanonical storage instead of silently coercing it.
- Alternate constructors exist only when they name a real storage or ownership policy.
- Generic scalar coercion factories such as `from_values(...)` are not introduced
  without a production caller and a distinct boundary responsibility.

## Validation structure

- Record constructors own field-local invariants.
- Solver templates own correlations between backend output and solver input.
- Reusable representation checks belong to the lowest dependency owner that defines the
  representation.
- One-call private validation helpers are not introduced solely to reduce McCabe
  complexity.
- Private transfer or evidence records are not introduced solely to split validation
  code.
- Private helpers represent reusable numerical operations, backend conversion, or
  coherent algorithm stages.
- A narrow complexity exemption is preferred over artificial helper decomposition when
  constructor invariants form one cohesive contract.

## Public contracts

- Public extension points do not require private handoff records.
- Backend handoff records are public when third-party nominal subclasses must construct
  them.
- Extension points remain closed when their implementation contract is intentionally
  private.
- Tests do not convert private helpers into de facto public APIs.

## Test construction

Tests should exercise public construction and behavior without manufacturing evidence
that the public contract cannot represent.

- Tests do not create evidence by changing arbitrary values in an otherwise valid record.
- Tests use named, independently derived mathematical cases and explicit contract
  boundaries.
- Invalid-state tests use explicit canonical constructor inputs or typed immutable
  backend probes.
- Tests may assert that ordinary field assignment raises `FrozenInstanceError` or
  `AttributeError`; failed assignment does not mutate the record.
- Tests do not bypass immutability or manufacture state through `object.__setattr__`,
  `object.__new__`, `dataclasses.replace`, generic override dictionaries, or equivalent
  mechanisms.
- Expected values are not changed to match observed production output.

## Enforcement

- AST inventories locate violations and support regression gates.
- AST inventories do not dictate ownership or architecture.
- Smells are remediated one capability or owner class at a time.
- Numerical operation order, exception classification, ownership, and protected
  provenance are preserved during remediation.
- Provenance-protected code is not refactored without explicit authorization.
