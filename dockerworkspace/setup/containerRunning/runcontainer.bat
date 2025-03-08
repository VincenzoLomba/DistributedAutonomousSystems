
@echo off
setlocal EnableDelayedExpansion
title Running ROS2 Docker Container

rem Setting up image and container names
set IMAGE_NAME=ros2_humble_image
set CONTAINER_NAME=ros2_humble
if "%~1" NEQ "" (
	if "%~2" NEQ "" (
		if "%~3" == "" (
			set IMAGE_NAME=%1
			set CONTAINER_NAME=%2
		)
	)
)
rem Retrieving current folder (if not already done)
if "%setup_path%"=="" set "setup_path=%~dp0.."

rem Checking if the container is already running and, in case, entering it
docker info >nul 2>nul
if %errorlevel% == 0 (
	for /f "tokens=*" %%i in ('docker ps --filter "name=%CONTAINER_NAME%" --format "{{.Names}}"') do set "RUNNING_CONTAINER=%%i"
	if defined RUNNING_CONTAINER (
		title Entering ROS2 Docker Container
		echo.
		echo ====================================================
		echo +          Entering ROS2 Docker Container          +
		echo ====================================================
		echo.
		echo Detected a Docker Container with name "%CONTAINER_NAME%" which is actually running... entering it^^!
		echo Good work fellow control engineer! ;^)
		docker exec -it %CONTAINER_NAME% /bin/bash
		exit /b 0
	)
)
rem Building ROS2 Docker Image (includes checking for Docker to be installed and running)
call "%setup_path%\imageBuilding\buildimage.bat" ros2_humble_image
if %errorlevel% NEQ 0 (exit /b 1)

echo.
echo ====================================================
echo +     Running ^& Entering ROS2 Docker Container     +
echo ====================================================
echo.

rem Defining VcXsrv configuration file path & VcXsrv default-to-use IP
set "xlog_file=%LOCALAPPDATA%\Temp\VCXSrv.0.log"
set "default_xip=host.docker.internal"

rem Checking for VcXsrv installation
set "xlaunch_path=C:\Program Files\VcXsrv\vcxsrv.exe"
set "xlaunch_path_x86=C:\Program Files (x86)\VcXsrv\vcxsrv.exe"
echo BE AWARE: this script relies on VcXsrv version 1.17.2.0
if exist "%xlaunch_path%" (
    set "xlaunch_exec=%xlaunch_path%"
) else if exist "%xlaunch_path_x86%" (
    set "xlaunch_exec=%xlaunch_path_x86%"
) else (
	echo ATTENTION: VcXsrv does not appear to be installed^^!
	echo =^> VcXsrv was not found in the folder: !xlaunch_path!
	echo =^> VcXsrv was also not found in the folder: !xlaunch_path_x86!
	echo.
	echo If that is the case, please download and install VcXsrv
	echo You can download VcXsrv from: https://vcxsrv.com/
	echo Alternatively, you can find the VcXsrv installer in the local folder: "containerRunning\VcXsrv"
	echo Install VcXsrv, and only after completing the installation, re-run this script^^!
	start "" "%setup_path%\containerRunning\VcXsrv\vcxsrv-64.1.17.2.0.installer.zip"
    exit /b 1
)

rem [Re]starting VcXsrv
tasklist | findstr /I "vcxsrv.exe" >nul
if %errorlevel% == 0 (
    echo VcXsrv already running, reloading it...
    taskkill /F /IM vcxsrv.exe >nul 2>&1
    timeout /t 2 >nul
) else (
	echo Now launching VcXsrv...
)
start "" "%xlaunch_exec%" :0 -ac -multiwindow -clipboard -logverbose 3

rem Reading VcXsrv logfile useful content (VcXsrv in-use IP and VcXsrv version)
echo Searching for the VcXsrv log file...
timeout /t 2 >nul
if not exist "%xlog_file%" (
	rem Unable to locate VcXsrv logfile...
	echo Warning: unable to locate and access VcXsrv log file^^!
	echo Indeed, VcXsrv log file was not found at the expected location: "!xlog_file!"
	echo Using default value for VcXsrv IP address: "%default_xip%"
	echo Warning: without accessing VcXsrv log file, unable to verify whether VcXsrv has been run correctly
	echo =^> You will have to check it manually by yourself^^!
	set "X_IP=%default_xip%"
) else (
	rem If available, detecting VcXsrv version...
	for /f "tokens=2 delims=:" %%A in ('findstr /c:"Release:" "%xlog_file%"') do (set "X_VERSION=%%A")
	if "!X_VERSION!" NEQ "" echo Detected VcXsrv version:!X_VERSION!
	rem Check if VcXsrv has been run correctly
	for /f "tokens=2 delims=:" %%A in ('findstr /c:"server error" "%xlog_file%"') do (set "X_LAUNCH_ERROR=%%A")
	if "!X_LAUNCH_ERROR!" NEQ "" (
		echo.
		echo Attention: unable to successfully run VcXsrv^^!
		echo Please, check for any errors related to VcXsrv, then re-run this script.
		echo Printing VcXsrv log trace:
		for /f "delims=" %%i in ('type "%xlog_file%"') do echo ^| %%i
		exit /b 1
	)
	rem VcXsrv logfile located, searching for VcXsrv version and in-use IP...
	rem First of all, locating IP and PORT line...
	findstr /C:"DISPLAY" "%xlog_file%" > "%xlog_file%.vcxsrv_extracted_line.log"
	for /f "usebackq delims=" %%A in ("%xlog_file%.vcxsrv_extracted_line.log") do (set "X_DISPLAY_LINE=%%A"	)
	if "!X_DISPLAY_LINE!" == "" (
		echo Warning: unable to retreive the VcXsrv in-use IP from the log file^^!
		echo Using default value for VcXsrv IP address: "%default_xip%"
		set "X_IP=%default_xip%"
	) else (
		rem Extracting IP and PORT...
		for /f "tokens=2 delims==" %%B in ('echo !X_DISPLAY_LINE! ^| findstr /i "DISPLAY="') do (set "TEMP_DISPLAY_LINE=%%B")
		rem Isolating IP...
		for /f "delims=:" %%C in ("!TEMP_DISPLAY_LINE!") do (
			set "X_IP=%%C"
			goto :continue
		)
		:continue
		echo Successfully retrieved VcXsrv in-use IP address: !X_IP!
	)
)

rem In case VcXsrv in-use IP is localhost (alias 127.0.0.1), setting it to the correct value for Docker container (alias host.docker.internal)
if "%X_IP%"=="127.0.0.1" set "X_IP=localhost"
if "%X_IP%" == "localhost" (
	set "X_IP=%default_xip%"
	echo VcXsrv is relying on localhost IP ^(alias 127.0.0.1^) =^> using "!X_IP!" as Docker env DISPLAY IP address
) else (
	echo Using "!X_IP!" as Docker env DISPLAY IP address
)

rem Running container...
echo.
echo Now running ROS2 Docker Container with name %CONTAINER_NAME%, then entering it...
echo Good work fellow control engineer! ;^)
set "DEV_NAME=user"
set "WS_NAME=syncworkspace"
set "WS_PATH=%setup_path%\..\%WS_NAME%"
for %%F in ("%WS_PATH%") do set "WS_PATH=%%~sF"
docker run --rm -it ^
	--privileged ^
	--env="DISPLAY=%X_IP%:0.0" ^
	--network=host ^
	--volume=%WS_PATH%:/home/%DEV_NAME%/%WS_NAME% ^
	--name %CONTAINER_NAME% %IMAGE_NAME%
exit /b %errorlevel%
