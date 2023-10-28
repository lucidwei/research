set PG_BIN=D:\Program Files\PostgreSQL\15\bin
set PG_USER=postgres
set PG_DATABASE=wgz_db

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "datetime=%%a"
set "timestamp=%datetime:~0,8%_%datetime:~8,4%"

set BACKUP_PATH=E:\BaiduNetdiskWorkspace\backups\pg_bkp_%timestamp%.tar

"%PG_BIN%\pg_ctl" start -D "D:\PostgreSQL\data"
"%PG_BIN%\pg_dump" -U %PG_USER% -w -F t -f "%BACKUP_PATH%" %PG_DATABASE% 



pause