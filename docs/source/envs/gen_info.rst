General Information
================================


.. exec::
    def print_actions_for_id(id):
        import gym
        import gym_craftingworld


        def print_json(arg):
            json_obj = str(arg)


            print('.. parsed-literal:: \n\n    %s\n\n\n' % json_obj)

        def prep_space(space):
            import gym
            if isinstance(space, gym.spaces.Dict):
                dct = {}
                for k in space.spaces:
                    dct[k] = prep_space(space.spaces[k])
                return dct
            else:
                return space



        envspec = gym.spec(id)


        print("")
        print("{}".format(id))
        print("=======================================")


        if 'docstr' in envspec._kwargs:
            print(envspec._kwargs['docstr'])


        env = gym.make(id)
        #action_space = prep_space(envspec._kwargs['action_space'])
        state_space = prep_space(env.observation_space)
        action_space = prep_space(env.action_space)

        print("------------------------")
        print("Observation Space")
        print("------------------------")
        print_json(state_space)


        print("------------------------")
        print("Action Space")
        print("------------------------")
        print_json(action_space)

        print("------------------------")
        print("Usage")
        print("------------------------")


        usage_str = '''.. code-block:: python

            import gym

            # Run a random agent through the environment
            env = gym.make("{}") # A {} env

            env.allow_gif_storage()

            obs = env.reset()
            done = False


            while not done:
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)


            env.reset()  # you must call reset after an episode in order to save the episode as a .gif

        '''.format(id,id,id)
        print(usage_str)



    ids = [
           'craftingworld-v0',
            'craftingworld-trees-axes-hammers',
           ]

    for i in ids:
        print_actions_for_id(i)

