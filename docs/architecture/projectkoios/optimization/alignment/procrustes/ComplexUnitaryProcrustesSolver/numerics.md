# Generic solver numerics

The base selects either `absolute + relative * largest` or
`max(absolute, relative * largest)`, fixes strict cutoff comparison semantics used by
families, and owns the residual coefficient
`max(64, 8(n + 2)) * epsilon`. It validates backend identity as coupled nonempty strings
for executed work or coupled absence for unexecuted work.
