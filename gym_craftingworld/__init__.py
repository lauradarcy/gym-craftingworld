from gym.envs.registration import register

register(id='craftingworld-v0',
         entry_point='gym_craftingworld.envs:CraftingWorldEnv',
         )