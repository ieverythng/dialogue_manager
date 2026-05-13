#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob

from setuptools import find_packages, setup

NAME = 'dialogue_manager'

setup(
    name=NAME,
    version='2.0.0',
    license='Apache-2.0',
    description='ROS4HRI-compatible dialogue manager',
    author='todo',
    author_email='todo@todo.todo',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + NAME, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['res/' + NAME]),
        ('share/' + NAME + '/launch', glob('launch/*.launch.py')),
        ('share/ament_index/resource_index/pal_configuration.' + NAME,
            ['config/' + NAME]),
        ('share/' + NAME + '/config', [
            'config/00-defaults.yml',
            'config/00-default_actions.yaml',
        ]),
    ],
    tests_require=['pytest'],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'start_manager = ' + NAME + '.start_manager:main'
        ],
    },
)
