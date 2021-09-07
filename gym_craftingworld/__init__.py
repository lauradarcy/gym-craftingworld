"""Register gym environments"""

from gym.envs.registration import register

register(id='craftingworld-v3',
         entry_point='gym_craftingworld.envs:CraftingWorldEnvRay',
         kwargs={'stacking':True, 'render_save_rate':10}
         )

register(id='craftingworldflat-v3',
         entry_point='gym_craftingworld.envs:CraftingWorldEnvFlat',
         kwargs={'stacking':True, 'render_save_rate':10}
         )
