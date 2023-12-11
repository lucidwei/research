WITH IndexClassification AS (
    SELECT 
        internal_id,
        CASE
            WHEN fund_fullname SIMILAR TO '%(国证2000指数|中证1000|中证2000|中小板300指数|上证380|上证中小|创业板指)%' THEN '小盘'
            WHEN fund_fullname SIMILAR TO '%(国证1000指数|中证500|中证800|中证200|上证中盘|中小板指|深证300价格|深证成份指数)%' THEN '中盘'
            WHEN fund_fullname SIMILAR TO '%(上证超级大盘|沪深300|上证180|巨潮100|上证50|中证100|深证100)%' THEN '大盘'        
            ELSE etf_type
        END AS predicted_etf_type
    FROM public.product_static_info
    WHERE product_type = 'fund'
      AND fund_fullname NOT LIKE '%债%'
      AND fund_fullname NOT LIKE '%QDII%'
      AND fund_fullname LIKE '%交易型开放式%'
      AND etf_type IS NULL
)

UPDATE public.product_static_info
SET etf_type = IndexClassification.predicted_etf_type
FROM IndexClassification
WHERE public.product_static_info.internal_id = IndexClassification.internal_id;
