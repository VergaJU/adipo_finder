from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="my_module",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.8",
)
