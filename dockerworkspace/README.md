## Hi fellow control engineer! <br>Let's work with ROS2 on a Windows Docker Container!

Hi fellow control engineer!<br>
Just download this whole [dockerworkspace][L0] folder (you can download it [from this link][L1]), run [runROS2container.bat][L2], follow its instructions and start working!<br>
**Notice that** the local folder "syncworkspace" and the container folder "/home/user/syncworkspace" will be synchronized in their content!<br>
<br>
Otherwise, here are the steps in case you want to set it up manually:
1. Install [Docker Desktop][L3].
2. Install [VcXsrv][L4], useful to make the Docker Container able to project images and videos in the windows system.
3. Build the required Docker Image with the command:<br>
   ```docker build "%DOCKERFILE_PATH%" --tag %IMAGE_NAME%```<br>
   where ```%DOCKERFILE_PATH%``` is the path of the folder which contains the [Dockerfile][L5] and where ```%IMAGE_NAME%``` is the Docker Image name to use.
4. Run VcXsrv with the command:<br>
   ```"%xlaunch_exec%" :0 -ac -multiwindow -clipboard -logverbose 3```<br>
   where ```%xlaunch_exec%``` is the path of the file ```vcxsrv.exe``` installed at step two.
5. Run the Docker Container with the command:<br>
   ```docker run --rm -it --privileged --env="DISPLAY=host.docker.internal:0.0" --network=host --volume=%WS_PATH%:/home/user/%WS_NAME% --name %CONTAINER_NAME% %IMAGE_NAME%```<br>
   where ```%WS_PATH%``` and ```%WS_NAME%``` are respectively the path of the in-windows folder and the name of the in-container folder that need to be synchronized (the container is assumed to have a user named ```user``` associated to an home folder ```/home/user```), ```%CONTAINER_NAME%``` is the name of the Docker Container and ```%IMAGE_NAME%``` is the name of Docker Image defined at point three.

Good work!

[L0]: https://github.com/VincenzoLomba/DistributedAutonomousSystems/tree/master/dockerworkspace
[L1]: https://downgit.github.io/#/home?url=https://github.com/VincenzoLomba/DistributedAutonomousSystems/tree/master/dockerworkspace
[L2]: https://github.com/VincenzoLomba/DistributedAutonomousSystems/blob/master/dockerworkspace/runROS2container.bat
[L3]: https://www.docker.com/products/docker-desktop
[L4]: https://vcxsrv.com/
[L5]: https://github.com/VincenzoLomba/DistributedAutonomousSystems/blob/master/dockerworkspace/setup/imageBuilding/Dockerfile
