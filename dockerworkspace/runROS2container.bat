
@echo off
set "setup_path=%~dp0setup"
call setup\containerRunning\runcontainer.bat
if %errorlevel% NEQ 0 (
	echo.
	pause
)
exit /b %errorlevel%