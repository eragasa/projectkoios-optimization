# CompletionEvidence implementation

`CompletionEvidence` is a frozen, slotted solver record. It groups exact completion,
rank, status, and cutoff array references between backend validation and public result
assembly. It is not exported by the façade.
