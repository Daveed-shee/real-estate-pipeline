-- Data quality check — how many nulls does each column have?

SELECT
    attname                                           AS column_name,
    null_frac,
    ROUND((null_frac * 2930)::numeric)                AS estimated_null_rows
FROM pg_stats
WHERE tablename = 'housing_raw'
ORDER BY null_frac DESC;