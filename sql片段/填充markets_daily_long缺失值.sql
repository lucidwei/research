UPDATE markets_daily_long mdl
SET
    product_static_info_id = psi.internal_id,
    code = psi.code
FROM
    product_static_info psi
WHERE
    mdl.product_name = psi.chinese_name
    AND (mdl.product_static_info_id IS NULL OR mdl.code = 'temp_value')
    AND psi.internal_id IS NOT NULL
    AND psi.chinese_name IN (
        SELECT chinese_name
        FROM product_static_info
        GROUP BY chinese_name
        HAVING COUNT(DISTINCT code) = 1
    );