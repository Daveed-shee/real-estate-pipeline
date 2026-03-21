-- What's the average sale price and price per sqft in each neighborhood?

SELECT
    neighborhood,
    COUNT(*)                                         AS num_sales,
    ROUND(AVG(saleprice))                            AS avg_sale_price,
    ROUND(AVG(saleprice / NULLIF(gr_liv_area, 0)))   AS avg_price_per_sqft,
    RANK() OVER (ORDER BY AVG(saleprice) DESC)        AS price_rank
FROM housing_raw
GROUP BY neighborhood
ORDER BY avg_sale_price DESC;