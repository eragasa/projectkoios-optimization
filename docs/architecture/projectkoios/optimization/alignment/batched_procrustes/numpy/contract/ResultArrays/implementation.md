# ResultArrays implementation

`ResultArrays` is a frozen, slotted dataclass in `contract.py`. It groups ten exact array
references between normalization and relational validation. It performs no copying or
mutation and is not exported by the façade.
