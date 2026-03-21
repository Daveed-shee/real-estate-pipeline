-- How did sale prices trend year over year?

SELECT
    yr_sold,
    COUNT(*)                              AS num_sales,
    ROUND(AVG(saleprice))                 AS avg_sale_price,
    ROUND(AVG(saleprice)) - LAG(ROUND(AVG(saleprice)))
        OVER (ORDER BY yr_sold)           AS yoy_change
FROM housing_raw
GROUP BY yr_sold
ORDER BY yr_sold;