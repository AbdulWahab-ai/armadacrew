# Cache stampede

Service: `auth-service`, `edge-gateway`, `payments-api` (authz cache).

## Signals

- `authz cache miss` rate > 40%
- Origin RPS spike with API RPS flat
- Redis CPU high, API p99 high

## Actions

1. Enable request coalescing on the cache key.
2. Serve stale-while-revalidate for 30s.
3. Do not flush redis in prod without approval — flush is a high-risk action.
4. Warm the JWKS and authz keys from a job, not from user traffic.
