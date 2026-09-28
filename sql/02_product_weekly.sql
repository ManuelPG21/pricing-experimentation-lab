-- Weekly product panel used for demand / price-elasticity estimation.
-- avg_price is the quantity-weighted average paid; it falls mechanically when bulk orders
-- (which get volume discounts) are large, biasing elasticity downwards.
-- list_price is the median line price that week, which is much less sensitive to order mix.
CREATE OR REPLACE VIEW product_weekly AS
SELECT
    stock_code,
    week,
    sum(quantity)                             AS units,
    sum(revenue) / sum(quantity)              AS avg_price,
    median(unit_price)                        AS list_price,
    count(DISTINCT invoice)                   AS orders,
    count(DISTINCT customer_id)               AS customers
FROM sales
WHERE country = 'United Kingdom'
GROUP BY stock_code, week;

-- Pareto view: revenue concentration by product.
CREATE OR REPLACE VIEW product_revenue AS
SELECT
    stock_code,
    any_value(description)                    AS description,
    sum(revenue)                              AS revenue,
    sum(quantity)                             AS units
FROM sales
GROUP BY stock_code;

-- Customer-level spend by period, used as the pre-period covariate for CUPED.
CREATE OR REPLACE VIEW customer_periods AS
SELECT
    customer_id,
    sum(CASE WHEN invoice_ts <  TIMESTAMP '2011-06-01' AND invoice_ts >= TIMESTAMP '2011-03-01' THEN revenue ELSE 0 END) AS pre_spend,
    sum(CASE WHEN invoice_ts >= TIMESTAMP '2011-06-01' AND invoice_ts <  TIMESTAMP '2011-09-01' THEN revenue ELSE 0 END) AS post_spend
FROM sales
WHERE customer_id IS NOT NULL
GROUP BY customer_id
HAVING pre_spend > 0 OR post_spend > 0;
