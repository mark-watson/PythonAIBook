---
type: Database Table
title: Customers
description: Anonymized customer dimension with loyalty tier and demographic segment attributes.
resource: postgres://retail-db/analytics/customers
tags: [customers, dimension, PII-anonymized, loyalty]
timestamp: 2026-05-20T08:00:00Z
owner: crm-team@example.com
row_count_estimate: 8_400_000
update_frequency: nightly batch
pii_classification: anonymized (no direct identifiers)
---

# Customers

The `customers` table provides a **privacy-safe** customer dimension.
All direct PII (name, email, address) has been removed; only derived
attributes and opaque identifiers are stored here.

## Schema

| Column | Type | Description |
|---|---|---|
| customer_id | INT | Opaque surrogate key (matches sales_events) |
| loyalty_tier | VARCHAR(16) | `Bronze`, `Silver`, `Gold`, `Platinum` |
| acquisition_channel | VARCHAR(32) | How the customer was first acquired |
| region | VARCHAR(32) | Broad geographic region (not city/zip) |
| age_band | VARCHAR(16) | `18-24`, `25-34`, `35-44`, `45-54`, `55+` |
| first_purchase_date | DATE | Date of very first transaction |
| lifetime_spend_usd | DECIMAL(14,2) | Cumulative historical spend |
| churn_risk_score | FLOAT | Model output, 0.0 (low) – 1.0 (high) |

## Usage Notes

- Only ~62 % of [sales_events](sales_events.md) rows have a non-NULL `customer_id`;
  the rest are anonymous sessions.
- `churn_risk_score` is refreshed weekly by the ML pipeline. Do not use it for
  real-time decisions.

## Related Metrics

- [metrics/customer_ltv](../metrics/customer_ltv.md)
