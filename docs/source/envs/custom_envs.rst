Custom Environments
================================
in order to create a custom environment, you need to import gym and gym_craftingworld, and then register your own version with the kwargs set as required

.. code-block:: python

    import gym
    import gym_craftingworld

    from gym.envs.registration import register
    register(id='craftingworld-MyCustomEnv',
             entry_point='gym_craftingworld.envs:CraftingWorldEnv',
             kwargs={'size': (20,5), 'object_ratios': (1, 0, 0, 0, 0, 0, 1, 1)}
             )

    env = gym.make('craftingworld-MyCustomEnv')


then just use the environment as usual.

further details about the init args are below:

.. autoclass:: gym_craftingworld.envs.craftingworld_env.CraftingWorldEnv
    :members:

    .. automethod:: __init__

