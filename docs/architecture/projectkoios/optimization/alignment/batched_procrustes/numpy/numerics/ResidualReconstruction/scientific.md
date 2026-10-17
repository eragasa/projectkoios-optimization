# ResidualReconstruction consumer needs

Consumers need a squared residual together with the cutoff that may have produced an
exact zero. They must persist both when zero-versus-roundoff distinction matters.

The class does not decide whether a residual is scientifically acceptable and is not
imported by consumers through the public façade.
