import setuptools

from iotdemo import __version__ as version

setuptools.setup(
    name="iotdemo",
    version=version,
    author="IoT demo makers group",
    description="IoT demo helper APIs",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.12',
    install_requires=[
        "openvino",
        "numpy>=2,<2.3.0",
    ],
    entry_points={
        'gui_scripts': [
            'iotdemo-motion-detector = iotdemo.tuning.motion:trampoline',
            'iotdemo-color-detector = iotdemo.tuning.color.color:main',
        ],
    },
)
