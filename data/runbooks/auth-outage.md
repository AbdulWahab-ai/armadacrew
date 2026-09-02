# Auth service outage

Service: `auth-service`
Typical causes: IdP latency (Okta), stale JWT `kid`, redis session store eviction.

## Symptoms

- Login failures, `jwt verify failed kid=legacy-2023`
- Elevated `error_rate` and `p99_ms` on `/v1/token` and `/v1/session`
- Downstream 401s on `payments-api` and `checkout-api`

## Immediate actions

1. Ack pager incident `inc-auth-yesterday` or the current key.
2. Confirm IdP latency; if issuer p99 > 1s, enable the cached JWKS fallback.
3. Rotate to the current signing `kid`; do not serve `legacy-2023`.
4. If redis-session is evicting, raise memory and disable non-auth keys on that cluster.
5. Prod restart of `auth-service` requires HITL approval.

## Postmortem notes

Yesterday's outage was a JWKS cache TTL bug after a key rotation. Customer impact: 18 minutes of login failures. Follow-up: dual-publish keys for 24h.
