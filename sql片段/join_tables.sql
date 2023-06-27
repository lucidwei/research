SELECT * FROM product_static_info INNER JOIN markets_daily_long 
ON product_static_info.internal_id = markets_daily_long.product_static_info_id
where product_type='index'