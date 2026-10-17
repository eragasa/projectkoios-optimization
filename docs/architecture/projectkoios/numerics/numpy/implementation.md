# Implementation

NumPy imports remain inside this namespace and concrete NumPy backend packages. Dependency-free numerical contracts do not import NumPy. Exact array and dtype checks are centralized in `validation`.
