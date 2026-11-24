@echo off
rem Drag a video onto this file. It plays it and writes what the RIFE filter did to rife-test.log.
cd /d "%~dp0"
mpv.com --msg-level=rife=v --log-file=rife-test.log %*
echo.
echo Log written to %~dp0rife-test.log
pause
