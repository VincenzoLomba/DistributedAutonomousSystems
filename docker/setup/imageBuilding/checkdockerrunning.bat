
@echo off
setlocal EnableDelayedExpansion
title Checking for Docker Installed And Running

rem Checking Docker Installation
where docker >nul 2>nul
if %errorlevel% NEQ 0 (
	echo Attention: Docker does not appear to be installed ^(^"docker^" not found in the PATH environment variable^)^^!
	echo If that is the case, please download and install Docker [Desktop].
	echo You can download Docker [Desktop] from: https://www.docker.com/products/docker-desktop
	echo Install Docker [Desktop], and only after completing the installation, re-run this script.
	start https://www.docker.com/products/docker-desktop
    exit /b 1
)

rem Checking for Docker actually running (in case not, running it)
docker info >nul 2>nul
set DOCKER_PATH=""
if %errorlevel% NEQ 0 (
    echo Warning: docker seems to be installed, but not actually running.
	set DOCKER_PATH="C:\Program Files\Docker\Docker\Docker Desktop.exe"
	if exist !DOCKER_PATH! (
		echo Docker Desktop installation found at !DOCKER_PATH!.
		echo Now attempting to run Docker Desktop ^(and waiting for it^)...
		start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
	) else (
		set DOCKER_PATH="C:\Program Files (x86)\Docker\Docker\Docker Desktop.exe"
		if exist !DOCKER_PATH! (
			echo Docker Desktop installation found at !DOCKER_PATH!.
			echo Now attempting to run Docker Desktop ^(and waiting for it^)...
			start "" "C:\Program Files (x86)\Docker\Docker\Docker Desktop.exe"
		)
	)
	if not exist !DOCKER_PATH! (
		echo Unable to locate Docker Desktop installation.
		echo Please, manually run Docker. Only then, re-run this script.
		exit /b 1
	)
)
set checking_max_attempts=5
set checking_attempts=0
set waiting_time_in_secs=3
:CHECK_DOCKER_RUNNING
docker info >nul 2>nul
if %errorlevel% == 0 (
	for /f "tokens=3 delims= " %%v in ('docker --version') do set DOCKER_VERSION=%%v
	set DOCKER_VERSION=!DOCKER_VERSION:,=!
	if %DOCKER_PATH%=="" (
		echo Detected running Docker ^(version !DOCKER_VERSION!^)
	) else (
		echo Docker now running ^(version !DOCKER_VERSION!^)^^!
	)
) else (
	timeout /t %waiting_time_in_secs% /nobreak >nul
	set /a checking_attempts+=1
    if %checking_attempts% LSS %checking_max_attempts% (
        set /a elaps=checking_attempts*waiting_time_in_secs
        echo Waiting for Docker Desktop to run ^(elapsed !elaps! seconds^)...
        goto CHECK_DOCKER_RUNNING
    ) else (
        echo Attention: unable to successfully run Docker Desktop.
        echo Please manually start Docker, then re-run this script.
        exit /b 1
    )
)
exit /b 0
