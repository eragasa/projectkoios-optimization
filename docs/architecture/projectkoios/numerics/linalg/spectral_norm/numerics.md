# Dense spectral-norm numerics

## Mathematical operation

For a matrix $A \in \mathbb{F}^{m \times n}$ with positive $m$ and $n$,

$$
\lVert A\rVert_2 = \sigma_1(A),
\qquad
\sigma_1(A) \geq \cdots \geq \sigma_{\min(m,n)}(A) \geq 0.
$$

`numpy.linalg.svdvals` computes the complete descending singular-value sequence. Index
zero therefore supplies the induced Euclidean matrix norm for real and complex square,
tall, and wide matrices.

One-row and one-column matrices reduce to Euclidean vector norms. Rank-deficient and
all-zero matrices follow the same complete-SVD path. The zero matrix returns exact
built-in `0.0`.

## Representation validation

`DenseMatrix` statically identifies NumPy arrays with `float64` or `complex128` element
types. NumPy's static array annotation does not encode exact runtime class,
dimensionality, nonempty shape, or entry finiteness. The implementation validates these
properties before SVD. This validation also prevents NumPy from treating a higher-rank
array as a stack of matrices or accepting an unsupported dtype through backend coercion.

The operation passes the validated array directly to NumPy. It does not copy, normalize,
convert, or mutate caller storage.

## Numerical and failure contract

The complete dense path has no caller tolerance, initial vector, iteration limit,
residual estimate, or fallback algorithm. NumPy `LinAlgError` denotes decomposition
nonconvergence and becomes `RuntimeError`. Other backend exceptions propagate unchanged.
Nonfinite singular-value output becomes `FloatingPointError`. The final
`MatrixNormABC.result()` boundary rejects a non-built-in, nonfinite, or negative scalar.

The owner implementation applies no explicit power-of-two scaling. Consumer-specific
scale-safe residual diagnostics remain separate capabilities.

## Verification arithmetic

Analytic fixtures define exact mathematical singular values independently of NumPy. Test
comparisons use the binary64 bound

$$
64\,\max(m,n)\,\operatorname{ulp}(1)\,\max(1,\sigma_1).
$$

The bound scales with problem dimension and norm magnitude. The factor 64 defines the
software-verification roundoff allowance; it is not a production tolerance or scientific
acceptance threshold. Separate fault-injection tests exercise NumPy nonconvergence and
nonfinite backend output because finite portable fixtures do not deterministically
produce those backend failures.
