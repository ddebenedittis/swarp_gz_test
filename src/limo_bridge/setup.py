from glob import glob

from setuptools import find_packages, setup

package_name = "limo_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Davide De Benedittis",
    maintainer_email="davide@cyber-fusion.io",
    description="Onboard Limo safety bridge: clamp, watchdog, e-stop, arena stop, namespaced relays.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "robot_bridge = limo_bridge.robot_bridge:main",
        ],
    },
)
