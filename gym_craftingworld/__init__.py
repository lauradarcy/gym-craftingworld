from gym.envs.registration import register

register(id='craftingworld-v3',
         entry_point='gym_craftingworld.envs:CraftingWorldEnvRay',
         kwargs={'stacking':True, 'render_flipping':False}
         )
