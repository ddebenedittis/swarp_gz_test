from glob import glob

from setuptools import find_packages, setup

package_name = "limo_real"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Davide De Benedittis",
    maintainer_email="davide@cyber-fusion.io",
    description="Laptop side of the real Limo Push-T setup.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mocap_relay = limo_real.mocap_relay:main",
            "amcl_relay = limo_real.amcl_relay:main",
            "fake_mocap = limo_real.fake_mocap:main",
            "operator = limo_real.operator:main",
            "pose_error = limo_real.pose_error:main",
            "pose_error_merge = limo_real.pose_error:merge_main",
            "measure_cmd_timeout = limo_real.measure.cmd_timeout:main",
            "measure_step_response = limo_real.measure.step_response:main",
            "measure_lidar_static = limo_real.measure.lidar_static:main",
            "measure_lidar_rate = limo_real.measure.lidar_rate:main",
        ],
    },
)
