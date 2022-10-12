"""
basic setup to start
"""
from setuptools import setup, find_packages

setup(name="medic",
	version='0.1dev',
	description="Model Error Detection in Cryo-EM",
	long_description=open('README.md').read(),
	author="Gabriella Reggiano",
	packages=find_packages(exclude=["medic_model"]),
	scripts=['detect_errors.py'],
	include_package_data=True,
	package_data={
		'DeepAccNet': ['deepAccNet/data/*.txt', 'deepAccNet/data/*.csv'],
	}
)
