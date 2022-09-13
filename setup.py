"""
basic setup to start
"""
from setuptools import setup, find_packages

setup(name="medic",
	version='0.1dev',
	description="Model Error Detection in Cryo-EM",
	long_description=open('README.md').read(),
	author="Gabriella Reggiano",
	packages=find_packages(where="medic"),
	package_dir={"":"medic"},
	package_data={"medic_model":"*"}
)
