@echo off
setlocal

:: 使用PowerShell获取当前日期的星期数，0 表示周日，6 表示周六
for /F "tokens=*" %%a in ('powershell -Command "(Get-Date).DayOfWeek.value__"') do set dayOfWeek=%%a

echo Day of the week: %dayOfWeek%

:: 检查当前是否为周六(6)或周日(0)
if "%dayOfWeek%"=="0" goto end
if "%dayOfWeek%"=="6" goto end

:: 设置日志文件路径和名称
set logPath=E:\BaiduNetdiskWorkspace\FICC_research\bat_scripts\logs

:: 检查日志路径是否存在，如果不存在则创建
if not exist "%logPath%" (
    echo Creating log directory...
    mkdir "%logPath%"
)

:: 使用PowerShell获取格式化的日期和时间
for /F "tokens=*" %%i in ('powershell -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "datetime=%%i"

:: 创建日志文件名
set "logFile=%logPath%\log_%datetime%.txt"

echo Log file will be: %logFile%

:: 运行Python脚本并记录输出到日志文件
echo Running the Python script...
:: 使用 PowerShell 运行 Python 脚本，并使用 Tee-Object 同时输出到文件和控制台
powershell -Command "D:\ProgramData\anaconda3\envs\touyan\python.exe E:\BaiduNetdiskWorkspace\FICC_research\daily_update.py *>&1 | Tee-Object -FilePath '%logFile%'; exit $LastExitCode"

if errorlevel 1 (
    echo Error running Python script. Error level: %errorlevel%
) else (
    echo Python script completed successfully.
)

:end
echo Script has finished.
pause
