
@echo off
setlocal EnableDelayedExpansion
title Building ROS2 Docker Image

echo.
echo ====================================================
echo +            Building ROS2 Docker Image            +
echo ====================================================
echo.

rem Retrieving current folder (if not already done)
if "%setup_path%"=="" set "setup_path=%~dp0.."

rem Checking for Docker to be installed and running
call "%setup_path%\imageBuilding\checkdockerrunning.bat" ros2_humble_image
if %errorlevel% NEQ 0 (exit /b 1)
title Building ROS2 Docker Image

rem Retrieving image name
set IMAGE_NAME=%1
if "%IMAGE_NAME%"=="" set IMAGE_NAME=ros2_humble_image
echo Now building image with name: %IMAGE_NAME%
echo.

rem Building Image
docker build "%setup_path%\imageBuilding" --tag %IMAGE_NAME%
exit /b %errorlevel%
