#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob

from setuptools import find_packages, setup

NAME = 'dialogue_manager'

setup(
    name=NAME,
    version='0.3.1',
    license='Apache-2.0',
    description='ROS4HRI-compatible dialogue manager',
    author='Severin Lemaignan',
    author_email='severin.lemaignan@iiia.csic.es',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + NAME, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['res/' + NAME]),
        ('share/' + NAME + '/launch', glob('launch/*.launch.py')),
        ('share/ament_index/resource_index/pal_system_module',
         ['module/' + NAME]),
        ('share/' + NAME + '/module', ['module/' + NAME + '_module.yaml']),
        ('share/ament_index/resource_index/pal_configuration.' + NAME,
            ['config/' + NAME]),
        ('share/' + NAME + '/config', ['config/00-defaults.yml']),
    ],
    tests_require=['pytest'],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'start_manager = ' + NAME + '.start_manager:main',
        ],
    },
)
