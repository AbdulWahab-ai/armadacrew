# Database saturation

Service targets: `checkout-db`, `billing-db`.

## Signals

- `sat_pct` > 85 for 8+ minutes
- Lock wait logs, `upstream timeout from checkout-db after 2000ms`
- API p99 climbs while RPS is flat (classic saturation, not load)

## Actions

1. Kill the top blocking query after confirming it is retry amplification.
2. Raise statement timeout to 1500ms on the API pool (not on analytics).
3. Shed reporting queries to the replica.
4. Do not fail over primary without approval — failover is a high-risk action.

## Rollback

Restore original pool size after `sat_pct` < 60 and p99 recovered for 15 minutes.
