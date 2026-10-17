# Generic solver numerics

The base fixes the additive cutoff `absolute + relative * largest`, strict cutoff
comparison semantics used by families, and the residual coefficient
`max(64, 8(n + 2)) * epsilon`. It validates backend identity as coupled nonempty strings
for executed work or coupled absence for unexecuted work.
