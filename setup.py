"""
basic setup to start
"""
from setuptools import setup, find_packages
import os

lib_folder = os.path.dirname(os.path.realpath(__file__))
long_description = open(os.path.join(lib_folder,'README.md')).read()

requirements_path = os.path.join(lib_folder, 'requirements.txt')
print(requirements_path)
install_requires = list()
if os.path.isfile(requirements_path):
	with open(requirements_path) as f:
		install_requires = f.read().splitlines()
print(install_requires)
setup(name="medic",
	version='0.1dev',
	description="Model Error Detection in Cryo-EM",
	long_description=long_description,
	author="Gabriella Reggiano",
	packages=find_packages(exclude=["medic_model"]),
	scripts=['detect_errors.py'],
	include_package_data=True,
	package_data={
		'DeepAccNet': ['deepAccNet/data/*.txt', 'deepAccNet/data/*.csv', 'models/NatComm_standard/*.pkl'],
	}
	install_requires=install_requires
)
