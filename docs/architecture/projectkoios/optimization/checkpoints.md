# Checkpoint contract

Schema version 1 uses canonical UTF-8 JSON with sorted keys, compact separators,
and no NaN or infinity. The envelope retains optimizer identity, canonical
payload text, schema version, and the SHA-256 digest of the exact payload bytes.
Restore rejects malformed JSON, unknown or missing fields, digest corruption,
unsupported schema, wrong optimizer identity, incompatible algorithm kind,
out-of-range or unrepresentable numeric values (including nested values), and
states that cannot be reproduced by deterministic replay. Such state failures
are reported as `CheckpointError`, including integer-to-float overflow.

No clock, hidden global random generator, filesystem location, or process state
is serialized. Therefore an uninterrupted engine and an engine restored at the
same ask/tell boundary produce exactly equal subsequent candidates and state.
