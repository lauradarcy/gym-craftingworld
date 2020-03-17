from gym.envs.registration import register

register(id='craftingworld-v0',
         entry_point='gym_craftingworld.envs:CraftingWorldEnv',
         )

register(id='craftingworld-v2',
         entry_point='gym_craftingworld.envs:CraftingWorldEnv',
         kwargs={'size': (5,5), 'object_ratios': (0, 1, 1, 0, 1, 0, 0, 0)}
         )