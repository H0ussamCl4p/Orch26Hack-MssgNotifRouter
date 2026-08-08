# Results

## Competition result

**#106 / 1983** globally — final score **70.9 / 100**. Solo, 24-hour build.

Scoring combined three dimensions: the agent's output (`output.csv`), the code,
and a 30-minute live technical interview with a judge who had read the submission.

## Frozen graded output

[`results/submission_output.csv`](../results/submission_output.csv) is the exact
110-row file that was graded. Nothing in the codebase writes to it; the pipeline's
own runs go to `results/output.csv` (gitignored).

SHA-256:

```
024a9a879c9d1b89b28697578c0fb85d2cbab5f5f665fa114b23b0190c26d056
```

Verify:

```bash
sha256sum results/submission_output.csv
```

## Metrics

Measured with `python -m router.evaluation` against the 30 labelled examples in
`sample_messages.csv`. **This is 30 examples — read every number as directional,
not exact.** One message moves accuracy by ~3 points.

| Metric | Value |
|---|---|
| action accuracy | 93% (28/30) |
| action macro-F1 | 0.93 |
| message_type accuracy | 87% (26/30) |
| message_type macro-F1 | 0.72 |
| evidence exact-match | 43% |
| evidence id precision | 52% |
| evidence id recall | 48% |
| calibration (Brier, action-correct) | 0.06 |
| retrieval recall (gold id in candidate pool) | 86% |
| API cost | $0 |

Calibration is monotone: action accuracy rises across confidence buckets
(86% → 92% → 100%).

## Why macro-F1 (0.72) trails accuracy (87%) on message_type

`spam` and `unknown` each have a support of **n = 1** in the 30 labels. A single
miss on either drives that class's F1 to 0.00, which the macro average weights
equally with well-supported classes. This is a small-sample artefact, not a
systemic classification failure — see the confusion matrix in the evaluation
output.

## Error analysis

[`results/eval_errors.csv`](../results/eval_errors.csv) lists every row where
action, message_type, or evidence disagreed with the gold label, with the
message text truncated for case-by-case reading. On this sample the dominant
error mode is evidence selection, not action/type classification.
