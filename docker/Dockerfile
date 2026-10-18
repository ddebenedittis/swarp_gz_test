# ROS 2 Jazzy + Gazebo Harmonic (Ubuntu 24.04, Python 3.12) for the Push-T Limo tests.
# Build with docker/build.bash (or `docker compose -f docker/docker-compose.yaml build`).
ARG BASE_IMAGE=osrf/ros:jazzy-desktop
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# ROS / Gazebo packages. ros-jazzy-ros-gz pulls Gazebo Harmonic through the gz-*-vendor
# packages, which also ship the gz.transport13 / gz.msgs10 Python bindings (on PYTHONPATH once
# /opt/ros/jazzy/setup.bash is sourced), so python3-gz-transport13 from packages.osrfoundation.org
# is not needed.
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    git \
    libegl1 \
    libgl1-mesa-dri \
    mesa-utils \
    python3-colcon-common-extensions \
    python3-pip \
    python3-venv \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-interfaces \
    ros-jazzy-rviz2 \
    ros-jazzy-xacro \
    sudo \
    xauth

# Python venv on top of the system site-packages (rclpy, numpy 1.26 from apt stay visible).
# numpy is pinned below 2 so the ROS C extensions built against numpy 1.x keep working.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv --system-site-packages ${VIRTUAL_ENV} \
 && ${VIRTUAL_ENV}/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && ${VIRTUAL_ENV}/bin/pip install --no-cache-dir "warp-lang>=1.15" "numpy>=1.26,<2" pytest
# The venv comes first on PATH. Its site-packages also go on PYTHONPATH, because ament_python
# console scripts get a /usr/bin/python3 shebang (colcon is a system package) and must still see
# torch/warp. /opt/swarp is the read-only bind mount of the sibling swarp repo.
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV PYTHONPATH=${VIRTUAL_ENV}/lib/python3.12/site-packages:/opt/swarp

# Non-root user matching the host UID/GID. Ubuntu 24.04 images ship an `ubuntu` user with
# UID/GID 1000: drop whatever already owns the requested IDs.
ARG USERNAME=ros
ARG UID=1000
ARG GID=1000
RUN if id -u ${UID} >/dev/null 2>&1; then userdel -r "$(id -nu ${UID})"; fi \
 && if getent group ${GID} >/dev/null; then groupdel "$(getent group ${GID} | cut -d: -f1)"; fi \
 && groupadd -g ${GID} ${USERNAME} \
 && useradd -m -s /bin/bash -u ${UID} -g ${GID} -G sudo,video ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/90-${USERNAME} \
 && chmod 0440 /etc/sudoers.d/90-${USERNAME}

COPY entrypoint.sh /entrypoint.sh
COPY bashrc.sh /etc/profile.d/zz-swarp-gz.sh
RUN chmod 0755 /entrypoint.sh \
 && echo 'source /etc/profile.d/zz-swarp-gz.sh' >> /etc/bash.bashrc

USER ${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR /home/${USERNAME}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
