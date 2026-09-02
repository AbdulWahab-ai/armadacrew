# Payments API high latency

Service: `payments-api`
Severity: SEV2 when p99 > 400ms for 10 minutes; SEV1 when p99 > 800ms or error rate > 5%.

## Symptoms

- Checkout conversion drop
- SLO burn alert `payments-api:latency:6h`
- Upstream timeouts from `checkout-db` and Stripe 429s

## Immediate actions

1. Ack the pager (`pager_ack`) and open a SEV ticket.
2. Confirm p99, p95, error_rate, and RPS vs baseline (last 60 minutes).
3. Check recent deploys; freeze rollouts if a canary is live.
4. Scale `payments-api` replicas +2 and enable load shedding on `/v1/charges` retries.
5. **Rolling restart in prod is high-risk** — requires human approval.
6. If Stripe is 429-ing, enable idempotency-key coalescing and pause duplicate capture jobs.

## Rollback

Revert the last chart change. Restore `redis-primary` circuit breaker to closed after error rate < 1%.

## Comms

Update the status page every 15 minutes. Customer-ops may see duplicate charges if captures retry.
