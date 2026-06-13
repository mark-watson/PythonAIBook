---
type: Database Table
title: Sales Events
description: Raw point-of-sale event stream capturing every transaction at the register level.
resource: postgres://retail-db/analytics/sales_events
tags: [sales, transactions, raw, events]
timestamp: 2026-06-01T09:00:00Z
owner: data-engineering@example.com
row_count_estimate: 2_500_000_000
update_frequency: real-time (CDC)
---

# Sales Events

The `sales_events` table is the **source of truth** for all transactional data
in the retail analytics platform. Every scan at a point-of-sale terminal writes
a record here within seconds via Change Data Capture.

## Schema

| Column | Type | Description |
|---|---|---|
| event_id | UUID | Unique identifier for the event |
| event_ts | TIMESTAMP | UTC timestamp of the transaction |
| store_id | INT | Foreign key to the store dimension |
| product_id | INT | Foreign key to [products](../tables/products.md) |
| customer_id | INT | Foreign key to [customers](../tables/customers.md), nullable |
| quantity | INT | Units sold (negative for returns) |
| unit_price_usd | DECIMAL(10,4) | Price per unit at time of sale |
| discount_pct | DECIMAL(5,2) | Discount percentage applied (0–100) |
| payment_method | VARCHAR(32) | `cash`, `credit`, `debit`, `gift_card` |

## Usage Notes

- **Partitioned** by `event_ts` (daily). Always filter on `event_ts` to avoid full scans.
- Negative `quantity` values represent returns; exclude them for revenue calculations.
- `customer_id` is NULL for roughly 38 % of transactions (anonymous cash sales).

## Lineage

Produced by the POS integration pipeline. Consumed by:
- [metrics/daily_revenue](../metrics/daily_revenue.md)
- [metrics/customer_ltv](../metrics/customer_ltv.md)
