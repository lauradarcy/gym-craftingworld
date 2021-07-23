Custom Environments
================================
in order to create a custom environment, you need to import gym and gym_craftingworld, and then register your own version with the kwargs set as required

.. code-block:: python

    import gym
    import gym_craftingworld

    from gym.envs.registration import register
    register(id='craftingworldMyCustomEnv-v3',
             entry_point='gym_craftingworld.envs:CraftingWorldEnvRay',
             kwargs={'stacking':True}
             )

    env = gym.make('craftingworldMyCustomEnv-v3')


then just use the environment as usual.

further details about the init args are below:

.. autoclass:: gym_craftingworld.envs.craftingworld_ray.CraftingWorldEnvRay
    :members:

    .. automethod:: __init__

