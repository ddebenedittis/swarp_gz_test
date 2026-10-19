"""Sim-to-sim system identification: the same scripted inputs in swarp and in Gazebo.

* :mod:`.experiments` defines the experiments (numpy only, shared by both sides);
* :mod:`.gz_runner` plays them in a running Gazebo sim (ROS node ``sysid_gz``);
* ``scripts/sysid_swarp.py`` plays them in swarp (host venv, imports swarp read-only);
* :mod:`.compare` turns both sides' ``out/sysid/*.npz`` into tables.
"""
