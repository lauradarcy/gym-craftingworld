"""Setup file for the gym-craftingworld PyPI package."""

import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym_craftingworld',
      version='0.1.9.2',
      author="Laura D\'Arcy",
      author_email="DArcyL@cardiff.ac.uk",
      description="A gym package for the 2d crafting multitask world",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/lauradarcy/gym-craftingworld",
      packages=setuptools.find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      install_requires=['gym', 'numpy', 'matplotlib', 'pillow']
      )
