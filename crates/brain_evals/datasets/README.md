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

One case per line, deserialised into `drift::DriftCase`.

```jsonc
# detect_topic_shift.v1 — drift detection golden set
{"id": "stays-on-topic-01", "turns": [{"role": "user", "text": "How do I index a Vec in Rust?"}, {"role": "assistant", "text": "Use square brackets, e.g. v[0]."}, {"role": "user", "text": "And how do I get its length?"}], "expect": {"shifted": false}, "notes": "follow-up on the same subject"}
{"id": "clear-shift-01", "turns": [{"role": "user", "text": "How do I index a Vec in Rust?"}, {"role": "assistant", "text": "Use square brackets."}, {"role": "user", "text": "Different