# `ScipyCsrSpectralNormControls` scientific boundary

`ScipyCsrSpectralNormControls` records software convergence and reproducibility inputs.
Its tolerance is an ARPACK and internal residual parameter, not a scientific error bar,
physical uncertainty, model discrepancy, or acceptance threshold.

Consumers own:

- selection between complete and iterative evaluation;
- justification of the numerical tolerance and iteration budget;
- persistence of the selected controls with consumer evidence;
- interpretation of approximation error; and
- human acceptance of results.

The seed controls deterministic start construction only. It does not turn backend output
into a scientifically validated random sample.
