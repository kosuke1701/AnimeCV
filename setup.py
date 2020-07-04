from setuptools import setup, find_namespace_packages

requirements = """numpy
pillow
requests
torch
torchvision
"""

setup(
    name="animecv",
    version="0.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements
)