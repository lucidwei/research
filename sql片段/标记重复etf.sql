WITH RankedResults AS (
    SELECT
        "public"."markets_daily_long"."product_static_info_id",
        ROW_NUMBER() OVER (
            PARTITION BY "public"."product_static_info"."fund_fullname"
            ORDER BY COUNT(*) DESC, LENGTH("public"."markets_daily_long"."product_name")
        ) AS rn
    FROM "public"."markets_daily_long"
    LEFT JOIN "public"."product_static_info" 
        ON "public"."markets_daily_long"."product_static_info_id" = "public"."product_static_info"."internal_id"
    WHERE "public"."markets_daily_long"."field" = '净流入额'
        AND "public"."product_static_info"."product_type" = 'fund'
    GROUP BY "public"."markets_daily_long"."product_name", "public"."product_static_info"."fund_fullname", "public"."product_static_info"."code", "public"."markets_daily_long"."product_static_info_id"
)
UPDATE
    public.product_static_info
SET
    etf_type = '重复'
WHERE
    internal_id IN (SELECT product_static_info_id FROM RankedResults WHERE rn != 1)
    AND fund_fullname LIKE '%交易型开放式%'
    AND product_type = 'fund';
-- 误标了很多重复，但也不是特别多。后续手动改回来一些