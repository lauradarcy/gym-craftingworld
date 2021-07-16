import gym
from gym import spaces
from gym.utils import seeding
import copy
import numpy as np
from numpy.core._multiarray_umath import ndarray

# from envs.custom_render import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import random
from gym_craftingworld.envs.coordinates import coord
import matplotlib.patches as mpatches
import os
from textwrap import wrap

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

PICKUPABLE = ['sticks', 'axe', 'hammer']
OBJECTS = ['sticks', 'axe', 'hammer', 'rock', 'tree', 'bread', 'house', 'wheat']

OBJECT_RATIOS = [1, 1, 1, 1, 1, 1, 1, 1]
OBJECT_PROBS = [x / sum(OBJECT_RATIOS) for x in OBJECT_RATIOS]

COLORS = [(110, 69, 39), (255, 105, 180), (100, 100, 200), (100, 100, 100), (0, 128, 0), (205, 133, 63), (197, 91, 97),
          (240, 230, 140)]
COLORS_N = [(0,0,0),(110, 69, 39), (255, 105, 180), (100, 100, 200), (100, 100, 100), (0, 128, 0), (205, 133, 63), (197, 91, 97),
          (240, 230, 140)]
COLORS_N_M = np.asarray(COLORS_N)
COLORS_rgba = [(110 / 255.0, 69 / 255.0, 39 / 255.0, .9), (255 / 255.0, 105 / 255.0, 180 / 255.0, .9),
               (100 / 255.0, 100 / 255.0, 200 / 255.0, .9), (100 / 255.0, 100 / 255.0, 100 / 255.0, .9),
               (0 / 255.0, 128 / 255.0, 0 / 255.0, .9), (205 / 255.0, 133 / 255.0, 63 / 255.0, .9),
               (197 / 255.0, 91 / 255.0, 97 / 255.0, .9), (240 / 255.0, 230 / 255.0, 140 / 255.0, .9)]
TASK_COLORS = ['red', 'green']
TASK_LIST = ['MakeBread', 'EatBread', 'BuildHouse', 'ChopTree', 'ChopRock', 'GoToHouse', 'MoveAxe', 'MoveHammer',
             'MoveSticks']

STATE_W = 21
STATE_H = 21

MAX_STEPS = 300


# TODO: check how multigoal worlds work in AI gym, does this affect use of done var, do we give a task to complete, etc
# TODO: maybe explicitly encode x and y as a feature or NOT convolutions - maybe to rbg encoding also?


class CraftingWorldEnvRay(gym.GoalEnv):
    """Custom Crafting environment that follows the gym interface pattern
    """

    metadata = {'render.modes': ['human', 'Non']}

    def __init__(self, store_gif=False, render_flipping=False, task_list=TASK_LIST,
                 selected_tasks=TASK_LIST, stacking=True):
        """
        change the following parameters to create a custom environment

        :param selected_tasks: list of tasks for the desired goal
        :param stacking: bool whether multiple tasks can be selected for desired goal
        :param store_gif: whether or not to store every episode as a gif in a /renders/ subdirectory
        :param render_flipping: set to true if only specific episodes need to be rendered
        :param task_list: list of possible tasks
        """
        self.seed()

        self.task_list = task_list

        self.selected_tasks = selected_tasks
        self.stacking = stacking
        pixel_w, pixel_h = STATE_W * 4, STATE_H * 4
        self.observation_space = spaces.Dict(dict(observation=spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h, 3),
                                                                         dtype=int),
                                                  desired_goal=spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h, 3),
                                                                          dtype=int),
                                                  achieved_goal=spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h, 3),
                                                                           dtype=int),
                                                  init_observation=spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h,
                                                                                                      3), dtype=int)))
        # self.observation_space = spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h, 3), dtype=int)

        self.observation_vector_space = spaces.Dict(dict(observation=spaces.Box(low=0, high=1,
                                                                                shape=(STATE_W, STATE_H,
                                                                                       len(OBJECTS) + 1 + len(
                                                                                           PICKUPABLE)),
                                                                                dtype=int),
                                                         desired_goal=spaces.Box(low=0, high=1,
                                                                                 shape=(1, len(self.task_list)),
                                                                                 dtype=int),
                                                         achieved_goal=spaces.Box(low=0, high=1,
                                                                                  shape=(1, len(self.task_list)),
                                                                                  dtype=int),
                                                         init_observation=spaces.Box(low=0, high=1,
                                                                                     shape=(STATE_W, STATE_H,
                                                                                            len(OBJECTS) + 1 + len(
                                                                                                PICKUPABLE)),
                                                                                     dtype=int)
                                                         ))

        self.desired_goal_vector = self.observation_vector_space.spaces['achieved_goal'].low

        self.achieved_goal_vector = np.zeros(shape=(1, len(self.task_list)), dtype=int)

        self.obs_one_hot, self.agent_pos = self.sample_state()

        # self.observation = {'observation': self.obs_one_hot, 'desired_goal': self.desired_goal_vector,
        #                     'achieved_goal': self.achieved_goal_vector}
        self.INIT_OBS_VECTOR = copy.deepcopy(self.obs_one_hot)
        self.INIT_OBS = self.render(self.INIT_OBS_VECTOR)
        self.observation_vector = {'observation': self.obs_one_hot, 'desired_goal': self.desired_goal_vector,
                                   'achieved_goal': self.achieved_goal_vector,
                                   'init_observation': self.INIT_OBS_VECTOR}
        # self.init_obs_one_hot = copy.deepcopy(self.obs_one_hot)
        self.init_observation_vector = copy.deepcopy(self.observation_vector)

        self.desired_goal = self.imagine_obs()
        self.observation = {'observation': self.render(self.obs_one_hot), 'desired_goal': self.desired_goal,
                            'achieved_goal': self.render(self.obs_one_hot),
                            'init_observation': self.INIT_OBS}
        self.init_observation = copy.deepcopy(self.observation)
        self.ACTIONS = [coord(-1, 0, name='up'), coord(0, 1, name='right'), coord(1, 0, name='down'),
                        coord(0, -1, name='left'), 'pickup', 'drop']

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # self.reward = self.calculate_rewards()

        self.store_gif = store_gif

        self.render_flipping = render_flipping
        self.env_id = None
        self.fig, self.ax, self.ims = None, None, None
        self.ep_no = 0
        self.step_num = 0
        if self.store_gif:
            self.allow_gif_storage()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, render_next=False):
        """
        reset the environment
        """
        # save episode as gif
        if self.store_gif is True and self.step_num != 0:
            # print('debug_final', len(self.ims))
            anim = animation.ArtistAnimation(self.fig, self.ims, interval=100000, blit=False, repeat_delay=1000)
            tasknums = '-'.join([str(i) for i in np.where(self.desired_goal_vector[0]==1)[0]])
            cpmleted = '-'.join([str(i) for i in np.where(self.achieved_goal_vector[0]==1)[0]])
            anim.save('renders/env{}/E{}({})_{}({}).gif'.format(self.env_id, self.ep_no, self.step_num, tasknums, cpmleted),
                      writer=animation.PillowWriter(), dpi=100)

        if self.render_flipping is True:
            self.store_gif = render_next

        if self.selected_tasks is not None:
            self.desired_goal_vector = np.zeros(shape=(1, len(self.task_list)), dtype=int)
            number_of_tasks = self.np_random.randint(len(self.selected_tasks)) + 1 if self.stacking is True else 1
            # tasks = self.np_random.sample(self.selected_tasks, k=number_of_tasks)
            task_idx = self.np_random.randint(0,len(self.selected_tasks)+1,size=number_of_tasks)
            for task in task_idx:
                self.desired_goal_vector[0][task] = 1
                # self.desired_goal_vector[0][self.task_list.index(task)] = 1
        else:
            self.desired_goal_vector = self.np_random.randint(2, size=(1, len(self.task_list)))

        self.achieved_goal_vector = np.zeros(shape=(1, len(self.task_list)), dtype=int)

        self.obs_one_hot, self.agent_pos = self.sample_state()

        self.INIT_OBS_VECTOR = copy.deepcopy(self.obs_one_hot)
        self.INIT_OBS = self.render(self.INIT_OBS_VECTOR)
        self.observation_vector = {'observation': self.obs_one_hot, 'desired_goal': self.desired_goal_vector,
                                   'achieved_goal': self.achieved_goal_vector,
                                   'init_observation': self.INIT_OBS_VECTOR}

        self.init_observation_vector = copy.deepcopy(self.observation_vector)

        self.desired_goal = self.imagine_obs()
        self.observation = {'observation': self.render(self.obs_one_hot), 'desired_goal': self.desired_goal,
                            'achieved_goal': self.render(self.obs_one_hot),
                            'init_observation': self.INIT_OBS}
        self.init_observation = copy.deepcopy(self.observation)

        # self.reward = self.calculate_rewards()

        if self.step_num != 0:  # don't increment episode number if resetting after init
            self.ep_no += 1

        self.step_num = 0

        if self.store_gif:
            # reset gif
            plt.close('all')
            # if self.fig is None:
            #     self.fig, self.ax = plt.subplots(1)
            # else:
            #     plt.clf()
            self.fig, self.ax = plt.subplots(1)
            self.ims = []
            self.__render_gif(reward=0)

        return self.observation

    def __object_list_to_state(self, object_dictionary, agent_pos):
        """
        produces a obs with one of each object
        :return obs: a sample obs
        :return agent_position: position of the agent within the obs
        """
        state = np.zeros(self.observation_vector_space.spaces['observation'].shape, dtype=int)
        for object_type, list_of_positions in object_dictionary.items():
            object_val = OBJECTS.index(object_type)
            for coordinate in list_of_positions:
                state[coordinate[0], coordinate[1]] = self.one_hot(obj=object_val, agent=False, holding=None)
        object_at_agent_pos, _, _ = CraftingWorldEnvRay.translate_one_hot(state[agent_pos.row, agent_pos.col])
        state[agent_pos.row, agent_pos.col] = self.one_hot(obj=object_at_agent_pos, agent=True, holding=None)
        final_goal = self.render(state=state)
        # return state, agent_pos
        return final_goal

    def __convert_item(self, object_dictionary, item_one, item_two=None, addl_item=None):
        if addl_item is not None:
            item = self.np_random.choice(object_dictionary[item_one] + object_dictionary[addl_item])
        else:

            self.np_random.permutation(object_dictionary[item_one])

            item = object_dictionary[item_one][0]
        object_dictionary[item_one].remove(item)
        if item_two is not None:
            object_dictionary[item_two].append(item)
        return object_dictionary


    def imagine_obs(self):
        init_objects = {obj: self.get_objects(code, self.init_observation_vector['observation']) for code, obj in
                        enumerate(OBJECTS)}
        agent_pos = self.agent_pos
        final_objects = copy.deepcopy(init_objects)

        tasks = {self.task_list[idx]: value for idx, value in enumerate(self.desired_goal_vector[0])}
        for key, value in tasks.items():
            if value == 1:
                if key == 'MakeBread':
                    final_objects = self.__convert_item(final_objects, 'wheat', 'bread')
                if key == 'EatBread':
                    final_objects = self.__convert_item(final_objects, 'bread')
                if key == 'ChopTree':
                    final_objects = self.__convert_item(final_objects, 'tree', 'sticks')
                if key == 'ChopRock':
                    final_objects = self.__convert_item(final_objects, 'rock')

        occupied_spaces = []
        for i in final_objects.values():
            occupied_spaces += i

        moving_tasks = {'MoveAxe': 'axe', 'MoveHammer': 'hammer', 'MoveSticks': 'sticks'}
        for key, value in moving_tasks.items():
            if key in tasks:
                if tasks[key] == 1:
                    current_location = self.np_random.choice(final_objects[value])
                    occupied = True
                    while occupied:
                        new_location = [self.np_random.randint(0, STATE_W - 1), self.np_random.randint(0, STATE_H - 1)]
                        if new_location not in occupied_spaces:
                            final_objects[value].remove(current_location)
                            occupied_spaces.remove(current_location)
                            final_objects[value].append(new_location)
                            occupied_spaces.append(new_location)
                            break

        for key, value in tasks.items():
            if value == 1:
                if key == 'BuildHouse':
                    final_objects = self.__convert_item(final_objects, 'sticks', 'house')

                if key == 'GoToHouse':
                    new_agent_pos = self.np_random.choice(final_objects['house'])
                    agent_pos = coord(new_agent_pos[0], new_agent_pos[1],
                                      STATE_W - 1, STATE_H - 1)

        # self.__object_list_to_state(final_objects, agent_pos)

        # self.__object_list_to_state(final_objects, agent_pos)

        # return final_objects, agent_pos
        return self.__object_list_to_state(final_objects, agent_pos)

    def step(self, action):
        """
        take a step within the environment

        :param action: integer value within the action_space range
        :return: observations, reward, done, debugging info
        """
        action_value = self.ACTIONS[action]
        self.step_num += 1

        # pull information from agent's current location
        current_cell = self.obs_one_hot[self.agent_pos.row, self.agent_pos.col]
        object_at_current_pos, _, what_agent_is_holding = CraftingWorldEnvRay.translate_one_hot(current_cell)

        # Execute one time step within the environment
        if action_value == 'pickup':
            if object_at_current_pos is None:
                pass  # nothing to pick up
            else:
                if what_agent_is_holding is not None:
                    pass  # print('already holding something')
                elif object_at_current_pos not in [0, 1, 2]:
                    pass  # print('can\'t pick up this object')
                else:
                    # print('picked up', CraftingWorldEnv.translate_state_code(obj_code))
                    self.obs_one_hot[self.agent_pos.row, self.agent_pos.col] = self.one_hot(agent=True,
                                                                                            holding=object_at_current_pos)

        elif action_value == 'drop':
            if what_agent_is_holding is None:
                pass  # nothing to drop
            else:
                if object_at_current_pos is not None:
                    pass  # print('can only drop items on an empty spot')
                else:
                    # print('dropped', CraftingWorldEnv.translate_state_code(holding_code+1))
                    self.obs_one_hot[self.agent_pos.row, self.agent_pos.col] = self.one_hot(obj=what_agent_is_holding,
                                                                                            agent=True)

        else:
            self.__move_agent(action_value)

        # task_success = self.eval_tasks()
        self.achieved_goal_vector = self.eval_tasks()
        self.observation_vector = {'observation': self.obs_one_hot, 'desired_goal': self.desired_goal_vector,
                                   'achieved_goal': self.achieved_goal_vector, 'init_observation': self.INIT_OBS_VECTOR}
        self.observation = {'observation': self.render(self.obs_one_hot), 'desired_goal': self.desired_goal,
                            'achieved_goal': self.render(self.obs_one_hot), 'init_observation': self.INIT_OBS}
        observation = self.observation
        # self.reward = self.calculate_rewards()
        reward = self.compute_reward(self.achieved_goal_vector, self.desired_goal_vector, None)
        # reward_lim = 0 if self.pos_rewards is False else np.sum(self.desired_goal_vector)
        done = True if self.step_num >= MAX_STEPS or reward == 1 else False

        # render if required
        if self.store_gif is True:
            if type(action_value) == coord:
                self.__render_gif(action_value.name, reward)
            else:
                self.__render_gif(action_value, reward)

        return observation, reward, done, {"task_success": self.achieved_goal_vector,
                                                          "desired_goal": self.desired_goal_vector,
                                                          "achieved_goal": self.achieved_goal_vector}

    def __move_agent(self, action):
        """
        updates the encoding of two locations in self.obs, the old position and the new position.

        first the function adds the coordinates together to get the new location,
        then we pull the contents of each location, including what the agent is holding

        the function performs a series of checks for special cases, i.e. what to do if moving onto a tree location

        then the function updates the encoding of the obs

        :param action: one of the movement actions, stored as a coordinate object. coordinate class makes it easier to ensure agent doesn't move outside the grid
        """
        new_pos = self.agent_pos + action

        if new_pos == self.agent_pos:  # agent is at an edge coordinate, so can't move in that direction
            return

        new_pos_encoding = self.obs_one_hot[new_pos.row, new_pos.col]
        object_at_new_pos, _, _ = CraftingWorldEnvRay.translate_one_hot(new_pos_encoding)

        current_pos_encoding = self.obs_one_hot[self.agent_pos.row, self.agent_pos.col]
        object_at_current_pos, _, what_agent_is_holding = CraftingWorldEnvRay.translate_one_hot(current_pos_encoding)

        if object_at_new_pos == 3:  # rock in new position
            if what_agent_is_holding != 2:  # agent doesn't have hammer
                # print('can\'t move, rock in way')
                return
            else:  # agent does have hammer
                object_at_new_pos = None  # remove rock

        elif object_at_new_pos == 4:  # tree in new position
            if what_agent_is_holding != 1:  # agent not holding axe
                # print('can\'t move, tree in way')
                return
            else:  # agent does have axe
                object_at_new_pos = 0  # turn tree into sticks

        elif object_at_new_pos == 0:  # sticks in new position
            if what_agent_is_holding == 2:  # agent has hammer
                object_at_new_pos = 6  # turn sticks into house

        elif object_at_new_pos == 7:  # wheat in new position
            if what_agent_is_holding == 1:  # agent has axe
                object_at_new_pos = 5  # turn wheat into bread

        elif object_at_new_pos == 5:  # bread in new position
            # print('removed bread')
            object_at_new_pos = None

        # update contents of new position
        self.obs_one_hot[new_pos.row, new_pos.col] = self.one_hot(obj=object_at_new_pos,
                                                                  agent=True, holding=what_agent_is_holding)

        # update contents of old position
        self.obs_one_hot[self.agent_pos.row, self.agent_pos.col] = self.one_hot(obj=object_at_current_pos, agent=False,
                                                                                holding=None)

        # update agent's location
        self.agent_pos = new_pos

    def render(self, state=None, mode='Non', tile_size=4):
        """

        :param mode: 'Non' returns the rbg encoding for use in __render_gif(). 'human' also plots for user.
        :param state: the observation needed to render. if None, will render current observation
        :param tile_size: the number of pixels per cell, default 4
        :return: rgb image encoded as a numpy array
        """
        if state is None:
            state = self.obs_one_hot

        height, width = state.shape[0], state.shape[1]
        objects, agents, holding = np.split(state,[len(OBJECTS),len(OBJECTS) + 1],axis=2)
        objects_n = np.concatenate((np.zeros((height,width,1),dtype=int),objects),axis=2)
        holding = np.concatenate((np.zeros((height, width, 1), dtype=int), holding), axis=2)

        new_state_h = np.argmax(holding, axis=2)

        img = np.tensordot(objects_n, COLORS_N_M,axes=1)
        img = np.repeat(img,4,axis=0)
        img = np.repeat(img,4,axis=1)
        a_x,a_y = self.agent_pos.t()
        img[a_x*4+1:a_x*4+3,a_y*4+1:a_y*4+3,:] = 255
        holding = np.max(new_state_h)
        if holding!=0:
            img[a_x * 4 + 2:a_x * 4 + 3, a_y * 4 + 1:a_y * 4 + 3] = COLORS_N[holding]

        if mode == 'human':
            fig2, ax2 = plt.subplots(1)
            ax2.imshow(img)
            fig2.show()

        return img

    def __render_gif(self, action_label=None, reward=404):
        img2 = self.render(mode='Non')
        im = plt.imshow(img2, animated=True)

        desired_goals = "\n".join(wrap(
            ', '.join([self.task_list[key] for key, value in enumerate(self.desired_goal_vector[0]) if value == 1]),
            50))
        achieved_goals = "\n".join(wrap(
            ', '.join([self.task_list[key] for key, value in enumerate(self.achieved_goal_vector[0]) if value == 1]),
            50))
        title_str = """
Episode {}: step {} - action choice: {}
Desired Goals: {}""".format(self.ep_no, self.step_num, action_label, desired_goals)

        bottom_text = "Achieved Goals: {}\nd_g: {}\na_g: {},   r: {}".format(achieved_goals, self.desired_goal_vector,
                                                                             self.achieved_goal_vector, reward)
        ttl = plt.text(0.00, 1.01, title_str, horizontalalignment='left',
                       verticalalignment='bottom', transform=self.ax.transAxes)
        txt = plt.text(0.00, -0.02, bottom_text, horizontalalignment='left',
                       verticalalignment='top', transform=self.ax.transAxes)
        plt.xticks([])
        plt.yticks([])
        patches = [mpatches.Patch(color=COLORS_rgba[i], label="{l}".format(l=OBJECTS[i])) for i in range(len(COLORS))]
        '''patches.append(mpatches.Patch(color='white', label="Tasks:"))
        tasks = [key for key,value in enumerate(self.desired_goal[0]) if value == 1]
        patches += [mpatches.Patch(color=TASK_COLORS[self.achieved_goal[0][idx]],
                                   label=self.task_list[idx]) for idx in tasks]'''
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)

        self.ims.append([im, ttl, txt])

    def sample_state(self):
        """
        produces a observation with one of each object
        :return obs: a sample observation
        :return agent_position: position of the agent within the observation
        """
        objects = [_ for _ in range(1, 10)]
        objects = [self.one_hot(i - 1) for i in objects]
        grid = objects + [[0 for _ in range(self.observation_vector_space.spaces['observation'].shape[2])]
                          for _ in range(STATE_W * STATE_H - len(objects))]
        self.np_random.shuffle(grid)

        state = np.asarray(grid, dtype=int).reshape(self.observation_vector_space.spaces['observation'].shape)

        agent_position = coord(int(np.where(np.argmax(state, axis=2) == 8)[0]),
                               int(np.where(np.argmax(state, axis=2) == 8)[1]),
                               STATE_W - 1, STATE_H - 1)

        return state, agent_position

    def eval_tasks(self):
        # print(self.obs_one_hot.shape, self.INIT_OBS_VECTOR.shape)

        objects, _, holding = np.split(self.obs_one_hot, [len(OBJECTS), len(OBJECTS) + 1], axis=2)
        objects = np.concatenate((np.zeros((STATE_H, STATE_W, 1), dtype=int), objects), axis=2)
        objects = np.argmax(objects,axis=2)
        holding = np.concatenate((np.zeros((STATE_H, STATE_W, 1), dtype=int), holding), axis=2)
        holding = np.argmax(holding,axis=2)
        init_objects, init_agent, init_holding = np.split(self.INIT_OBS_VECTOR, [len(OBJECTS), len(OBJECTS) + 1], axis=2)
        init_objects = np.concatenate((np.zeros((STATE_H, STATE_W, 1), dtype=int), init_objects), axis=2)
        init_objects = np.argmax(init_objects,axis=2)

        f_obj_count = np.asarray([np.count_nonzero(objects==i) for i in range(1,9)])
        i_obj_count = np.asarray([np.count_nonzero(init_objects==i) for i in range(1,9)])
        delta_obj = i_obj_count-f_obj_count

        make_bread = delta_obj[7] > 0
        eat_bread = delta_obj[7]+delta_obj[5] > 0
        build_house = delta_obj[6] < 0
        chop_tree = delta_obj[4] > 0
        chop_rock = delta_obj[3] > 0

        house_locations = np.argwhere(objects==7)
        go_to_house = any((self.agent_pos.t() == x).all() for x in house_locations)

        initial_stick_locations = np.argwhere(init_objects==1)
        final_stick_locations = np.argwhere(objects==1)
        if not any((initial_stick_locations == x).all() for x in final_stick_locations) or len(final_stick_locations)==0:
            if objects[initial_stick_locations[0][0],initial_stick_locations[0][1]] == 7:
                # there is a house where the sticks were, this doesnt count as moving!
                move_sticks = False
            else:
                # if the agent is in OG stick spot while holding sticks, false, else sticks have been moved
                move_sticks = not (any((self.agent_pos.t() == x).all() for x in initial_stick_locations) and np.max(holding) == 1)
        else:
            if not chop_tree:
                move_sticks = False
            else:
                tree_init_spot = np.argwhere(init_objects==5)
                tree_new_object = objects[tree_init_spot[0][0],tree_init_spot[0][1]]
                if tree_new_object == 1 or tree_new_object == 7:
                    # if sticks are still where the tree was or were converted to house on the spot, it doesnt count
                    move_sticks = False
                else:
                    # if the agent is in OG tree spot while holding sticks, false, else sticks have been moved
                    move_sticks = not (any((self.agent_pos.t() == x).all() for x in tree_init_spot) and np.max(
                        holding) == 1)

        if delta_obj[1] == 0:
            move_axe = not np.array_equal(np.argwhere(objects==2),np.argwhere(init_objects==2))
        else:
            move_axe = not np.array_equal(np.argwhere(init_objects==2),np.asarray([self.agent_pos.t()]))

        if delta_obj[2] == 0:
            move_hammer = not np.array_equal(np.argwhere(objects==3),np.argwhere(init_objects==3))
        else:
            move_hammer = not np.array_equal(np.argwhere(init_objects==3),np.asarray([self.agent_pos.t()]))

        task_success = np.asarray([[make_bread, eat_bread, build_house, chop_tree, chop_rock, go_to_house, move_axe, move_hammer,
             move_sticks]],dtype=int)
        return task_success

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.sum(np.square(desired_goal - achieved_goal)) == 0:
            return 1
        else:
            return -1

    def calculate_rewards(self, desired_goal=None, achieved_goal=None, initial_goal=None):
        if self.binary_rewards is True:
            if desired_goal is None:
                desired_goal = self.observation_vector['desired_goal']
            if achieved_goal is None:
                achieved_goal = self.observation_vector['achieved_goal']
            if np.sum(np.square(desired_goal - achieved_goal)) == 0:
                return 1
            else:
                return -1
        if desired_goal is None:
            desired_goal = self.observation['desired_goal']
        if achieved_goal is None:
            achieved_goal = self.observation['achieved_goal']
        if initial_goal is None:
            initial_goal = self.init_observation['achieved_goal']
        error = np.sqrt(np.sum(np.square(desired_goal - achieved_goal)))
        if self.pos_rewards is True:
            initial_error = np.sqrt(np.sum(np.square(desired_goal - initial_goal)))
            return -error / initial_error
        return -error

    def get_objects(self, code, state):
        """
        returns the locations for a particular type object within a obs

        :param code: the code of the object, which is the index of the object within the one-hot encoding
        :param state: the obs to search in
        :return: a list of locations where the object is $[[i_1,j_1],[i_2,j_2],...,[i_n,j_n]]$
        """
        code_variants = [code]
        if code < 3:
            code_variants.append(code + 9)
        locations = []
        for i in range(STATE_W):
            for j in range(STATE_H):
                for c in code_variants:
                    if state[i, j, c] == 1:
                        locations += [[i, j]]
        return locations

    def allow_gif_storage(self, store_gif=True):
        """
        turns on or off gif storage, this is a separate function because it create a new subdirectory,
        so wanted the user to have to explicitly call this function

        :param store_gif: a boolean, set to true to turn on gif storage.
        """
        self.store_gif = store_gif
        if self.store_gif is True:
            self.env_id = self.np_random.randint(0, 1000000)

            os.makedirs('renders/env{}'.format(self.env_id), exist_ok=False)
            self.fig, self.ax = plt.subplots(1)
            self.ims = []  # storage of step renderings for gif
            self.__render_gif()

    def one_hot(self, obj=None, agent=False, holding=None):
        row = [0 for _ in range(self.observation_vector_space.spaces['observation'].shape[2])]
        if obj is not None:
            row[obj] = 1
        if agent:
            row[8] = 1
        if holding is not None:
            row[holding + 9] = 1
        return row

    @staticmethod
    def translate_one_hot(one_hot_row):
        object_at_location = np.argmax(one_hot_row[:len(OBJECTS)]) if one_hot_row[:len(OBJECTS)].any() == 1 else None
        holding = np.argmax(one_hot_row[len(OBJECTS) + 1:]) if one_hot_row[len(OBJECTS) + 1:].any() == 1 else None
        agent = one_hot_row[len(OBJECTS)]
        return object_at_location, agent, holding
