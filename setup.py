from setuptools import find_packages,setup
from typing import List


def req(filename:str)->List[str]:
    requirements = []
    with open(filename) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements ]

    if '-e .' in requirements:
        requirements.remove('-e .')

    return requirements

setup(
    name = 'mlopsproject_face_recognition',
    version = '0.0.1',
    author = 'Benito Bosco Sinukaban',
    author_email= 'benitobosco8@gmail.com',
    packages = find_packages(),
    install_requires = req('requirements.txt')
)