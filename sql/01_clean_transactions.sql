-- Clean sales lines: drop cancellations, returns, non-product codes and missing prices.
CREATE OR REPLACE VIEW sales AS
SELECT
    Invoice                                   AS invoice,
    StockCode                                 AS stock_code,
    trim(Description)                         AS description,
    Quantity                                  AS quantity,
    Price                                     AS unit_price,
    Quantity * Price                          AS revenue,
    CAST(InvoiceDate AS TIMESTAMP)            AS invoice_ts,
    date_trunc('week', CAST(InvoiceDate AS TIMESTAMP)) AS week,
    CAST("Customer_ID" AS BIGINT)             AS customer_id,
    Country                                   AS country
FROM raw
WHERE Invoice NOT LIKE 'C%'
  AND Quantity > 0
  AND Price > 0
  AND regexp_matches(StockCode, '^[0-9]{5}[A-Za-z]?$');
