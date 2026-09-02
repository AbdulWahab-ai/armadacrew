# Pod crash-loop / restart storm

## Diagnose

- `pod memory working_set_bytes` climbing to cgroup limit
- Restart count > 3 in 10 minutes
- Readiness failing while liveness still passes (traffic blackhole)

## Actions

1. Snapshot logs before they rotate.
2. Raise memory limit 25% as a temporary mitigation (not a fix).
3. Disable the newest feature flag if it coincides with the deploy.
4. **Rolling restart of prod is high-risk** and needs a human gate.
5. After restart, watch error budget for 20 minutes before unfreezing deploys.
