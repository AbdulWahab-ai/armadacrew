# Payments capture / duplicate charge failures

Service: `payments-api`
Related KB: duplicate charges, refund policy.

## Symptoms

- `duplicate capture suspected payment_intent=...`
- Stripe 429 on `/v1/charges`
- Customers reporting double billing (often ~$720 enterprise invoices)

## Actions

1. Freeze the capture worker.
2. Inspect idempotency keys; coalescing must be on.
3. For confirmed duplicates, open a billing ticket and route to support.
4. Refunds over $500 USD require human approval.
5. Do not retry captures until the 429 rate window closes.

## Validation

CRM should show `duplicate_charge_usd` on the customer record. Match `payment_intent` before refunding.
