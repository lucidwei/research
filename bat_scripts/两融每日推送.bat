@echo off
:: 强制utf8，解析中文路径
chcp 65001 >nul
:: 使用PowerShell获取当前日期的星期数，1 表示周一，7 表示周日
for /F "tokens=*" %%a in ('powershell -Command "(Get-Date).DayOfWeek.value__"') do set dayOfWeek=%%a

echo Day of the week: %dayOfWeek%

:: 检查当前是否为周六(7)或周日(0)
if "%dayOfWeek%"=="0" goto end
if "%dayOfWeek%"=="7" goto end

echo Running the VBScript...
cscript //Nologo "E:\BaiduNetdiskWorkspace\FICC_research\bat_scripts\RunVBA.vbs"

echo Running the Python script...
D:\ProgramData\anaconda3\envs\touyan\python.exe "E:\BaiduNetdiskWorkspace\FICC_research\bat_scripts\两融保存图片.py"

echo Scripts have finished executing.

pause
:end