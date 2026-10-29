# NumpyClosedLoopPolarFrameTransportDefinition

`NumpyClosedLoopPolarFrameTransportDefinition` owns one immutable exact-array input:

- finite `complex128` frames with shape `(M, ambient, rank)`;
- `M >= 1` and `0 < rank <= ambient`;
- finite `complex128` endpoint sewing map with shape `(ambient, ambient)`;
- required `ClosedLoopPolarFrameTransportPrincipalPhaseBranch`; and
- exact `ComplexUnitaryProcrustesTolerance`.

`from_arrays` makes defensive byte-backed immutable copies.
`from_immutable_arrays` accepts only canonical immutable storage. Frame columns must be
orthonormal within the retained maximum-absolute-entry validation tolerance. The sewing
map is not assumed unitary and receives no domain meaning here. `negative_pi_open`
selects `(-pi, pi]`; `positive_pi_open` selects `[-pi, pi)`.
