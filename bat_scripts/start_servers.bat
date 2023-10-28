@echo on
echo Starting PostgreSQL...
pg_ctl start -D "D:\PostgreSQL\data"

echo Starting Metabase...
start "" "D:\metabase\metabase.jar" > nul

echo Starting Flask...
call "D:\anaconda\Scripts\activate.bat" "D:\anaconda\envs\FICC_research"
cd "E:\BaiduNetdiskWorkspace\FICC_research"
gunicorn --bind 0.0.0.0:8001 picture_server:app

echo Done.

