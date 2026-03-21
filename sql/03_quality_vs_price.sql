-- Does overall quality rating actually predict sale price?

SELECT
    overall_qual,
    COUNT(*)                  AS num_homes,
    ROUND(AVG(saleprice))     AS avg_sale_price,
    ROUND(MIN(saleprice))     AS min_price,
    ROUND(MAX(saleprice))     AS max_price
FROM housing_raw
GROUP BY overall_qual
ORDER BY overall_qual DESC;