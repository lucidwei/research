WITH IndexClassification AS (
    SELECT 
        internal_id,
	        fund_fullname,
        etf_type AS current_etf_type,
        CASE
            WHEN fund_fullname SIMILAR TO '%(红利|高股息|低波)%' THEN '红利'
            WHEN fund_fullname SIMILAR TO '%(消费)%' THEN '消费'   
			WHEN fund_fullname SIMILAR TO '%(上海金|黄金)%' THEN '黄金'   
	WHEN fund_fullname SIMILAR TO '%(大盘)%' THEN '大盘'   
	WHEN fund_fullname SIMILAR TO '%(中盘)%' THEN '中盘'   
	WHEN fund_fullname SIMILAR TO '%(小盘|国证2000)%' THEN '小盘'   
	WHEN fund_fullname SIMILAR TO '%(创业板)%' THEN '创业板'   
	WHEN fund_fullname SIMILAR TO '%(科创板)%' THEN '科创板'
	WHEN fund_fullname SIMILAR TO '%(科创创业)%' THEN '科创创业'
            ELSE etf_type
        END AS predicted_etf_type
    FROM public.product_static_info
    WHERE product_type = 'fund'
      AND fund_fullname NOT LIKE '%债%'
      AND fund_fullname NOT LIKE '%QDII%'
      AND fund_fullname LIKE '%交易型开放式%'
      AND etf_type IS NULL
)
---SELECT * FROM IndexClassification;
UPDATE public.product_static_info
SET etf_type = IndexClassification.predicted_etf_type
FROM IndexClassification
WHERE public.product_static_info.internal_id = IndexClassification.internal_id;