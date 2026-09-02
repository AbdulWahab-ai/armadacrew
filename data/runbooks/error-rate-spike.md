# Error-rate spike (5xx / timeout)

Applies to any service behind `edge-gateway`.

## Diagnose

- Query `error_rate` and `p99_ms` together. Isolated CPU without 5xx is not this runbook.
- Search logs for `DeadlineExceeded`, `circuit breaker`, `retry exhausted`.
- Correlate `trace_id` across the caller and callee.

## Mitigate

1. Shed retry storms (cap client retries at 1).
2. Open the circuit on the worst dependency for 60s.
3. Scale the callee, not the caller, if saturation (`sat_pct`) is high.
4. Page the on-call for that dependency.

## Do not

Do not restart prod pods until saturation and retry storms are ruled out. Restarts without shedding recreate the thundering herd.
