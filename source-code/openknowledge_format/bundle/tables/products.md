---
type: Database Table
title: Products
description: Product catalog containing SKU-level attributes, category hierarchy, and supplier information.
resource: postgres://retail-db/analytics/products
tags: [products, catalog, reference, dimension]
timestamp: 2026-05-15T12:00:00Z
owner: merchandising@example.com
row_count_estimate: 120_000
update_frequency: daily batch (midnight UTC)
---

# Products

The `products` table is the **canonical product catalog** for the retail platform.
It is a slowly changing dimension (SCD Type 2) — historical versions of product
records are retained so that sales reports can reflect the price and category
at the time of the transaction.

## Schema

| Column | Type | Description |
|---|---|---|
| product_id | INT | Surrogate key |
| sku | VARCHAR(32) | Natural business key (globally unique) |
| name | VARCHAR(256) | Display name |
| category_l1 | VARCHAR(64) | Top-level category (e.g., `Electronics`) |
| category_l2 | VARCHAR(64) | Sub-category (e.g., `Mobile Phones`) |
| brand | VARCHAR(128) | Brand name |
| cost_usd | DECIMAL(10,4) | Wholesale cost at time of record |
| list_price_usd | DECIMAL(10,4) | Recommended retail price |
| supplier_id | INT | Foreign key to supplier dimension |
| is_active | BOOLEAN | Whether the SKU is currently sold |
| valid_from | DATE | SCD validity start |
| valid_to | DATE | SCD validity end (NULL = current) |

## Usage Notes

- Join to [sales_events](sales_events.md) on `product_id` **and** filter
  `valid_from <= event_ts::date AND (valid_to IS NULL OR valid_to >= event_ts::date)`
  to get the correct historical version.
- Use `is_active = TRUE` for current inventory analysis.
