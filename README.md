# gym-craftingworld

[![Documentation Status](https://readthedocs.org/projects/gym-craftingworld/badge/?version=latest)](https://gym-craftingworld.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/lauradarcy/gym-craftingworld.svg?branch=master)](https://travis-ci.com/lauradarcy/gym-craftingworld)

This is a (work-in-progress) gym package for a 2D crafting environment first described in the [Plan Arithmetic: Compositional Plan Vectors for Multi-Task Control](https://arxiv.org/abs/1910.14033), published in NeurIPS 2019.
The mechanics of this environment are adapted from this [code](https://github.com/cdevin/craftingworld).
You can read the documentation [here](https://gym-craftingworld.readthedocs.io/).

## Usage

To store each episode as a gif, call `env.allow_gif_storage()`, which will store each episode in the `/renders/` subdirectory.
