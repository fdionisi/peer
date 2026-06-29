# Golden datasets

Each file is a versioned golden dataset for one invocation action, named
`<action>.<version>.jsonl`. The version lives in the filename so a revision is a
new, reviewable file rather than an in-place mutation of an existing one. Files
hold inputs and *expectations* only; run results (scores, model id, prompt hash)
are recorded separately in a report, because outputs are stochastic and the
dataset must stay stable across runs.

Format is JSON Lines: one case per line. Blank lines and lines beginning with
`#` are ignored, so a file may carry a header comment.

## detect_topic_shift

Deserialised into `drift::DriftCase`. Each case is a conversation (a sequence
of turns) plus whether the topic shifted and, if so, at which turn index the
new topic begins. The two error modes are not equally expensive: a spurious
split fragments a coherent conversation and writes a noisy summary into the
recall index, so the dataset is deliberately biased toward stay cases to keep
precision honest.

## tool_search_invocation

Deserialised into `tool_search::ToolSearchCase`. Each case is a single user
message plus the tool the model should call first. The positive case must
request a capability the model provably lacks (e.g. filesystem access), so
discovery is the only rational path; a request the model can satisfy natively
gives it no reason to search.