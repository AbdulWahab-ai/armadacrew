# Billing FAQ

**Q: I was charged twice.**
A: Duplicate captures can happen when Stripe 429s collide with client retries. We reverse the duplicate. Amounts over $500 need a supervisor.

**Q: How long until a refund lands?**
A: 5–7 business days on the original payment method.

**Q: Can I get a refund on a consumed API quota?**
A: Only if the quota was unusable due to a documented SEV1.

Never put card PAN, SSN, or street address in outbound email. That is PII and trips the email approval gate.
