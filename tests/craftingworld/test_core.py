import gym
import random
from gym.envs.registration import register

testenv = gym.make('craftingworld-v2')


def test_get_size():
    size = (random.randint(1, 15), random.randint(1, 15))
    register(id='craftingworldsizetest-v0',
             entry_point='gym_craftingworld.envs:CraftingWorldEnv',
             kwargs={'size': size})
    testenv = gym.make('craftingworldsizetest-v0')
    assert (testenv.num_rows, testenv.num_cols) == size
