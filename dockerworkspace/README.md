## Hi fellow control engineer! <br>Let's work with ROS2 on a Windows Docker Container!

Hi fellow control engineer!<br>
Just run [runROS2container.bat][L1], follow its instructions and start working!<br>
**Notice that** the local folder "syncworkspace" and the container folder "/home/user/syncworkspace" will be synchronized in their content!<br>
<br>
Otherwise, here are the steps IF you want do the setup manually:
1. Install [Docker Desktop][L2].
2. Install [VcXsrv][L3], useful to make a Docker Container able to project images and videos in the windows system.
3. Build the required Docker Image with the command:<br>
   ```docker build "%DOCKERFILE_PATH%" --tag %IMAGE_NAME%```<br>
   where ```%DOCKERFILE_PATH%``` is the path of the folder which contains the [Dockerfile][L4] and where ```%IMAGE_NAME%``` is the Docker Image name to use.
4. Run VcXsrv with the command:<br>
   ```"%xlaunch_exec%" :0 -ac -multiwindow -clipboard -logverbose 3```<br>
   where ```%xlaunch_exec%``` is the path of the file ```vcxsrv.exe``` installed at step two.
5. Run the Docker Container with the command:<br>
   ```docker run --rm -it --privileged --env="DISPLAY=host.docker.internal:0.0" --network=host --volume=%WS_PATH%:/home/user/%WS_NAME% --name %CONTAINER_NAME% %IMAGE_NAME%```<br>
   where ```%WS_PATH%``` and ```%WS_NAME%``` are respectively the path of the in-windows folder and the name of the in-container folder that need to be synchronized (the container is assumed to have a user named ```user``` associated to an home folder ```/home/user```), ```%CONTAINER_NAME%``` is the name of the Docker Container and ```%IMAGE_NAME%``` is the Docker Image name defined at point 3.

[L1]: https://github.com/VincenzoLomba/DistributedAutonomousSystems/blob/master/dockerworkspace/runROS2container.bat
[L2]: https://www.docker.com/products/docker-desktop
[L3]: https://vcxsrv.com/
[L4]: https://github.com/VincenzoLomba/DistributedAutonomousSystems/blob/master/dockerworkspace/setup/imageBuilding/Dockerfile
