# Gym Craftingworld

[![Downloads](https://img.shields.io/pypi/dm/gym-craftingworld)](https://pypi.org/project/gym-craftingworld/)
[![Lines](https://img.shields.io/tokei/lines/github/lauradarcy/gym-craftingworld)](https://github.com/lauradarcy/gym-craftingworld)
[![Documentation Status](https://readthedocs.org/projects/gym-craftingworld/badge/?version=latest)](https://gym-craftingworld.readthedocs.io/en/latest/?badge=latest)

This is a (work-in-progress) gym package for a 2D crafting environment.
This environment is based off one first described in [Plan Arithmetic: Compositional Plan Vectors for Multi-Task Control](https://arxiv.org/abs/1910.14033), published in NeurIPS 2019, and the mechanics are adapted from the corresponding code at [code](https://github.com/cdevin/craftingworld).

You can read the documentation [here](https://gym-craftingworld.readthedocs.io/).

## Environment description

This environment consists of a two-dimensional top-down, grid-based world consisting of 4x4 pixel cells.
The environment contains seven object types of object:

* `Tree`
* `Rock`
* `Logs`
* `Wheat`
* `Bread`
* `Hammer`
* `Axe`

Different objects are represented by differently coloured 4x4 blocks, while the agent is represented by a 2x2 white pixel block centered within the 4x4 cell.

The agent's actions are descrete and consist of six possible choices:

* `Up`
* `Down`
* `Left`
* `Right`
* `Pickup`
* `Drop`

Logs, hammers, and axes can be picked up by agent.
Trees and rocks block the agent's movement.

The environment consists of nine skills:

* `ChopTree`
* `BuildHouse`
* `MakeBread`
* `EatBread`
* `BreakRock`
* `GoToHouse`
* `MoveAxe`
* `MoveHammer`
* `MoveSticks`

A task is defined by a list of skills. For example, [`ChopTree`, `BuildHouse`].

The quantities and positions of each object are randomly selected at each reset, which occurs between episodes.

## Rendering

To store each episode as a gif, call `env.allow_gif_storage()`, which will store each episode in the `/renders/` subdirectory.
