# Point-in-time delayed-label integrity

Model performance is only meaningful when every outcome was genuinely observable at the stated
evaluation time. A historical report must not improve merely because labels arrived after that
report's `as_of` timestamp.

## Ingestion contract

Predictions use `(model_id, model_version, entity_id)` as their idempotency key. Labels use
`entity_id`. Replaying the exact same immutable record is accepted as an idempotent no-op and
returns an accepted count of zero. Reusing either key with a different score, window, timestamp
or label fails with HTTP 409 and rolls back the whole batch.

All prediction and label timestamps must include a timezone. They are normalized to UTC before
persistence. Scores must be finite probabilities, labels must be binary and identifiers must be
non-empty.

The current label table is immutable by design. A corrected outcome therefore needs a future,
explicit versioned correction workflow rather than an in-place rewrite that silently changes old
performance evidence.

## Point-in-time evaluation

The performance endpoint accepts an optional `as_of` timestamp and label-coverage floor:

```http
POST /v1/models/credit-risk/versions/2026.10/performance/october/evaluate
     ?as_of=2026-11-15T00:00:00Z
     &minimum_label_coverage=0.80
```

The evaluator:

1. rejects predictions timestamped after `as_of`;
2. rejects labels observed before their prediction, because the outcome was already known;
3. excludes labels observed after `as_of`;
4. reports `prediction_count`, `joined_count` and `label_coverage`;
5. rejects the report before persistence when coverage is below the configured floor.

When `as_of` is omitted, the current UTC time is used. `minimum_label_coverage` defaults to zero
for API compatibility; a production deployment should choose a policy from the outcome-maturity
curve and decision risk rather than copy a universal threshold.

## Boundary and limitations

This control prevents known temporal leakage and silent replay mutation inside this service. It
does not prove that an upstream label is correct, that the entity join is semantically valid, or
that missing labels are random. Selection bias can remain even at high coverage. Repeated
historical evaluations replace the report for the same model/version/window; preserving every
as-of vintage would require a versioned performance-run key and retention policy.

For real deployment, monitor coverage by segment and label delay, define a versioned correction
workflow, and retain the exact `as_of` and policy settings with every promotion decision.
