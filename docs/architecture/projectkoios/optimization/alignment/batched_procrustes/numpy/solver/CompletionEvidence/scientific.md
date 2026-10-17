# CompletionEvidence consumer needs

Consumers never import `CompletionEvidence`. Encapsulation prevents incomplete solver
construction state from becoming a downstream scientific contract. Consumers receive
only the validated public result and apply their own interpretation there.
