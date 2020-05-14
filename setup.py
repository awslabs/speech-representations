import os
from setuptools import setup, find_packages

data_files = []

setup(
    name="speech-reps",
    version="0.1",

    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},

    # include data files
    data_files=data_files,

    install_requires=[
        'numpy',
        'gluonnlp~=0.9.1',
        'soundfile',
        'kaldi_io'
    ],

    entry_points = {
        'console_scripts': [
            'speech-reps=speech_reps.cmds:main',
            ],
    }

)