@echo off

:: 使用PowerShell获取当前日期的星期数，1 表示周一，7 表示周日
for /F "tokens=*" %%a in ('powershell -Command "(Get-Date).DayOfWeek.value__"') do set dayOfWeek=%%a

echo Day of the week: %dayOfWeek%

:: 检查当前是否为周六(7)或周日(0)
if "%dayOfWeek%"=="0" goto end
if "%dayOfWeek%"=="7" goto end

cscript //Nologo RunVBA.vbs
D:\ProgramData\anaconda3\envs\touyan\python.exe 两融每日推送.py
