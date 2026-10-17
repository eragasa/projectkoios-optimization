# Scalar solver consumer needs

Consumers use the scalar family for one prepared matrix pair. They own whether the pair
represents physical frames, whether the tolerance is scientifically adequate, and how a
rank-deficient result affects downstream work. The solver owns none of those meanings.
