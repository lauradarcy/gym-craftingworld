Custom Environments
================================
in order to create a custom environment, you need to import gym and gym_craftingworld, and then register your own version with the kwargs set as required

.. code-block:: python

    import gym
    import gym_craftingworld

    from gym.envs.registration import register
    register(id='craftingworldMyCustomEnv-v0',
             entry_point='gym_craftingworld.envs:CraftingWorldEnv',
             kwargs={'size': (20,5), 'fixed_goal': ['MakeBread','EatBread','BuildHouse','MoveAxe'],
                 'tasks_to_ignore':['MoveAxe','MoveHammer','MoveSticks']}
             )

    env = gym.make('craftingworldMyCustomEnv-v0')


then just use the environment as usual.

further details about the init args are below:

.. autoclass:: gym_craftingworld.envs.craftingworld_env.CraftingWorldEnv
    :members:

    .. automethod:: __init__

