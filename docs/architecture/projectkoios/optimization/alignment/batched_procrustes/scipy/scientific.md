# SciPy batched Procrustes consumer needs

No current consumer requires a SciPy-specific batched Procrustes backend. Consumers use
the supported NumPy façade and retain all frame, unit, topology, threshold, persistence,
and scientific-acceptance policy outside Optimization.

A future request must identify a concrete representation or execution need that NumPy
does not satisfy. Scientific terminology alone is not a reason to add backend code, and
no consumer policy may leak into a future SciPy implementation.
