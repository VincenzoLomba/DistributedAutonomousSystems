
FROM ros:humble-ros-base

ENV DEV_NAME=user
ENV ROS_DISTRO=humble
ENV WS_NAME=syncworkspace 

# Group and user creation
RUN groupadd -g 1000 ${DEV_NAME} && \
    useradd -d /home/${DEV_NAME} -s /bin/bash -m ${DEV_NAME} -u 1000 -g 1000 && \
    usermod -aG sudo ${DEV_NAME} && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Creating the bridge folder
RUN mkdir -p /home/${DEV_NAME}/${WS_NAME}

# Installing NAV2
RUN apt-get update -q && \
    apt-get install -yq \
        ros-${ROS_DISTRO}-rqt-graph \
        ros-${ROS_DISTRO}-navigation2 \
        ros-${ROS_DISTRO}-nav2-bringup \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${DEV_NAME}/.bashrc

# Installing Utils
RUN apt update -q && \
    apt install -yq \
        gedit \
        iputils-ping \
        nano \
        net-tools \
        python3-pip \
        vim \
        xterm \
        less \
        x11-apps \
        htop \
        btop && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
COPY ./requirements.txt /home/${DEV_NAME}
RUN pip3 install -r /home/${DEV_NAME}/requirements.txt 

# Setting up useful aliases
RUN echo 'PROMPT_DIRTRIM=1' >> /home/${DEV_NAME}/.bashrc
RUN echo 'export ROS_DOMAIN_ID=100' >> /home/${DEV_NAME}/.bashrc

# Finalize
WORKDIR /home/${DEV_NAME}/${WS_NAME}
USER ${DEV_NAME}