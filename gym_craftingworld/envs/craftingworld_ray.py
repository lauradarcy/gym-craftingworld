import gym
from gym import spaces
from gym.utils import seeding
# import copy
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gym_craftingworld.envs.coordinates import Coord
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
COLORS_N = [(0, 0, 0), (110, 69, 39), (255, 105, 180), (100, 100, 200), (100, 100, 100), (0, 128, 0), (205, 133, 63),
            (197, 91, 97),
            (240, 230, 140)]
COLORS_H = np.asarray([[145, 186, 216], [0, 150, 75], [155, 155, 55]])

COLORS_N_M = np.asarray(COLORS_N)
COLORS_M = np.asarray(COLORS)
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

    def __init__(self, size=(STATE_W,STATE_H), max_steps=MAX_STEPS, store_gif=False, render_flipping=False, task_list=TASK_LIST,
                 selected_tasks=TASK_LIST, number_of_tasks=None, stacking=True, reward_style=None):
        """
        change the following parameters to create a custom environment

        :param selected_tasks: list of tasks for the desired goal
        :param stacking: bool whether multiple tasks can be selected for desired goal
        :param store_gif: whether or not to store every episode as a gif in a /renders/ subdirectory
        :param render_flipping: set to true if only specific episodes need to be rendered
        :param task_list: list of possible tasks
        """
        self.seed()
        if reward_style is None:
            self.compute_reward = self.compute_reward_equal
        else:
            self.compute_reward = self.compute_reward_subset
        self.STATE_W, self.STATE_H = size
        self.MAX_STEPS = max_steps
        self.task_list = task_list
        self.selected_tasks = selected_tasks
        self.number_of_tasks = number_of_tasks if number_of_tasks is not None else len(self.selected_tasks)
        if self.number_of_tasks > len(self.selected_tasks):
            self.number_of_tasks = len(self.selected_tasks)

        self.stacking = stacking
        pixel_w, pixel_h = self.STATE_W * 4, self.STATE_H * 4
        self.observation_space = spaces.Dict(dict(observation=spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h, 3),
                                                                         dtype=int),
                                                  desired_goal=spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h, 3),
                                                                          dtype=int),
                                                  achieved_goal=spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h, 3),
                                                                           dtype=int),
                                                  init_observation=spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h,
                                                                                                      3), dtype=int)))

        self.observation_vector_space = spaces.Dict(dict(observation=spaces.Box(low=0, high=1,
                                                                                shape=(self.STATE_W, self.STATE_H,
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
                                                                                     shape=(self.STATE_W, self.STATE_H,
                                                                                            len(OBJECTS) + 1 + len(
                                                                                                PICKUPABLE)),
                                                                                     dtype=int)
                                                         ))

        self.desired_goal_vector = self.observation_vector_space.spaces['achieved_goal'].low

        self.achieved_goal_vector = np.zeros(shape=(1, len(self.task_list)), dtype=int)

        self.obs_one_hot, self.agent_pos = None, None
        self.obs_image = None
        self.INIT_OBS_VECTOR = None
        self.INIT_OBS = None
        self.observation_vector = None
        # self.init_observation_vector = None

        self.desired_goal = None
        self.observation = None
        # self.init_observation = None

        self.ACTIONS = [Coord(-1, 0, name='up'), Coord(0, 1, name='right'), Coord(1, 0, name='down'),
                        Coord(0, -1, name='left'), 'pickup', 'drop']

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.store_gif = store_gif

        self.render_flipping = render_flipping
        self.env_id = None
        self.fig, self.ax1, self.ax2, self.ims = None, None, None, None
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
            anim = animation.ArtistAnimation(self.fig, self.ims, interval=100000, blit=False, repeat_delay=1000)
            tasknums = '-'.join([str(i) for i in np.where(self.desired_goal_vector[0] == 1)[0]])
            cpmleted = '-'.join([str(i) for i in np.where(self.achieved_goal_vector[0] == 1)[0]])
            anim.save(
                'renders/env{}/E{}({})_{}({}).gif'.format(self.env_id, self.ep_no, self.step_num, tasknums, cpmleted),
                writer=animation.PillowWriter(), dpi=100)

        if self.render_flipping is True:
            self.store_gif = render_next

        number_of_tasks = self.np_random.randint(self.number_of_tasks) + 1 if self.stacking is True else 1
        self.desired_goal_vector = np.zeros(shape=(1, len(self.task_list)), dtype=int)
        task_idx = np.arange(len(self.selected_tasks))
        self.np_random.shuffle(task_idx)
        for idx in task_idx[0:number_of_tasks]:
            self.desired_goal_vector[0][self.task_list.index(self.selected_tasks[idx])]=1

        self.achieved_goal_vector = np.zeros(shape=(1, len(self.task_list)), dtype=int)

        self.obs_one_hot, self.agent_pos = self.sample_state()

        self.INIT_OBS_VECTOR = self.obs_one_hot.copy()

        self.observation_vector = {'observation': self.obs_one_hot, 'desired_goal': self.desired_goal_vector,
                                   'achieved_goal': self.achieved_goal_vector,
                                   'init_observation': self.INIT_OBS_VECTOR}

        # self.init_observation_vector = copy.deepcopy(self.observation_vector)

        self.desired_goal = self.imagine_obs()
        self.obs_image = self.render(self.obs_one_hot)
        self.INIT_OBS = self.obs_image.copy()
        self.observation = {'observation': self.obs_image, 'desired_goal': self.desired_goal,
                            'achieved_goal': self.obs_image,
                            'init_observation': self.INIT_OBS}

        # self.init_observation = copy.deepcopy(self.observation)

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
            self.fig = plt.figure()
            self.ax1 = self.fig.add_subplot(1, 2, 1)
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            self.ims = []
            self.__render_gif(state_image=self.obs_image, reward=0)

        return self.observation

    def imagine_obs(self):
        # OBJECTS = ['sticks', 'axe', 'hammer', 'rock', 'tree', 'bread', 'house', 'wheat']
        # TASK_LIST = ['MakeBread', 'EatBread', 'BuildHouse', 'ChopTree', 'ChopRock', 'GoToHouse', 'MoveAxe',
        # 'MoveHammer',
        # 'MoveSticks']
        final_state = self.INIT_OBS_VECTOR.copy()
        if self.desired_goal_vector[0][0] == 1:  # MakeBread
            # wheat_loc = np.unravel_index(np.flatnonzero(final_state[:,:,7] == 1),
                                         # final_state[:, :, 7].shape)
            wheat_loc = np.where(final_state[:,:,7] == 1)
            final_state[wheat_loc[0][0], wheat_loc[1][0], 7] = 0
            final_state[wheat_loc[0][0], wheat_loc[1][0], 5] = 1
        if self.desired_goal_vector[0][1] == 1:  # EatBread
            # bread_loc = np.unravel_index(np.flatnonzero(final_state[:, :, 5] == 1),
                                         # final_state[:, :, 5].shape)
            bread_loc = np.where(final_state[:,:,5] == 1)
            which_bread = self.np_random.randint(len(bread_loc[0]))
            final_state[bread_loc[0][which_bread],bread_loc[1][which_bread],5] = 0
        if self.desired_goal_vector[0][3] == 1:  # ChopTree
            # tree_loc = np.unravel_index(np.flatnonzero(final_state[:, :, 4] == 1),
                                         # final_state[:, :, 4].shape)
            tree_loc = np.where(final_state[:,:,4] == 1)
            final_state[tree_loc[0][0], tree_loc[1][0], 4] = 0
            final_state[tree_loc[0][0], tree_loc[1][0], 0] = 1
        if self.desired_goal_vector[0][8] == 1:  # MoveSticks
            # stick_loc = np.unravel_index(np.flatnonzero(final_state[:, :, 0] == 1),
                                         # (STATE_W, STATE_H))
            stick_loc = np.where(final_state[:,:,0] == 1)
            which_stick = self.np_random.randint(len(stick_loc[0]))
            # unoccupied_spaces = np.unravel_index(np.flatnonzero(np.sum(final_state[:,:,:9],axis=2)),
                                                 # (STATE_W,STATE_H))
            # print(np.sum(final_state[:, :, :9], axis=2).shape)
            unoccupied_spaces = np.where(np.add.reduce(final_state[:,:,:9],axis=2) == 0)
            # unoccupied_spaces = np.where(np.sum(final_state[:,:,:9],axis=2)==0)
            # print(np.add.reduce(final_state[:,:,:9],axis=2).shape)
            which_spot = self.np_random.randint(len(unoccupied_spaces[0]))
            final_state[stick_loc[0][which_stick], stick_loc[1][which_stick], 0] = 0
            final_state[unoccupied_spaces[0][which_spot], unoccupied_spaces[1][which_spot], 0] = 1
        if self.desired_goal_vector[0][2] == 1:  # BuildHouse
            # stick_loc = np.unravel_index(np.flatnonzero(final_state[:, :, 0] == 1),
                                         # (STATE_W, STATE_H))
            stick_loc = np.where(final_state[:,:,0] == 1)
            which_stick = self.np_random.randint(len(stick_loc[0]))
            final_state[stick_loc[0][which_stick], stick_loc[1][which_stick], 0] = 0
            final_state[stick_loc[0][which_stick], stick_loc[1][which_stick], 6] = 1
        if self.desired_goal_vector[0][4] == 1:  # ChopRock
            # rock_loc = np.unravel_index(np.flatnonzero(final_state[:, :, 3] == 1),  (STATE_W, STATE_H))
            rock_loc = np.where(final_state[:,:,3] == 1)
            final_state[rock_loc[0][0], rock_loc[1][0], 3] = 0
        if self.desired_goal_vector[0][5] == 1:  # GoToHouse
            # house_loc = np.unravel_index(np.flatnonzero(final_state[:, :, 6] == 1),
            #                              (STATE_W, STATE_H))
            house_loc = np.where(final_state[:,:,6] == 1)
            which_house = self.np_random.randint(len(house_loc[0]))
            final_state[house_loc[0][which_house], house_loc[1][which_house], 8:] = final_state[self.agent_pos.row,
                                                                                    self.agent_pos.col, 8:]
            final_state[self.agent_pos.row,self.agent_pos.col,8:]= 0
        if self.desired_goal_vector[0][6] == 1:  # MoveAxe
            # axe_loc = np.unravel_index(np.flatnonzero(final_state[:, :, 1] == 1),  (STATE_W, STATE_H))
            axe_loc = np.where(final_state[:,:,1] == 1)
            # unoccupied_spaces = np.unravel_index(np.flatnonzero(np.sum(final_state[:, :, :8], axis=2)),
            #                                      (STATE_W, STATE_H))
            unoccupied_spaces = np.where(np.add.reduce(final_state[:, :, :8], axis=2) == 0)
            # unoccupied_spaces = np.where(np.sum(final_state[:,:,:8],axis=2)==0)
            which_spot = self.np_random.randint(len(unoccupied_spaces[0]))
            final_state[axe_loc[0][0], axe_loc[1][0], 1] = 0
            final_state[unoccupied_spaces[0][which_spot], unoccupied_spaces[1][which_spot], 1] = 1
        if self.desired_goal_vector[0][7] == 1:  # MoveHammer
            # hammer_loc = np.unravel_index(np.flatnonzero(final_state[:, :, 2] == 1),  (STATE_W, STATE_H))
            # print(final_state[:, :, 2].shape)
            hammer_loc = np.where(final_state[:,:,2] == 1)
            # unoccupied_spaces = np.unravel_index(np.flatnonzero(np.sum(final_state[:, :, :8], axis=2)),
                                                 # (STATE_W, STATE_H))
            unoccupied_spaces = np.where(np.add.reduce(final_state[:, :, :8], axis=2) == 0)
            # unoccupied_spaces = np.where(np.sum(final_state[:,:,:8],axis=2)==0)
            which_spot = self.np_random.randint(len(unoccupied_spaces[0]))
            final_state[hammer_loc[0][0], hammer_loc[1][0], 2] = 0
            final_state[unoccupied_spaces[0][which_spot], unoccupied_spaces[1][which_spot], 2] = 1

        return self.render(final_state)

    def step(self, action):
        """
        take a step within the environment

        :param action: integer value within the action_space range
        :return: observations, reward, done, debugging info
        """
        action_value = self.ACTIONS[action]
        self.step_num += 1

        # Execute one time step within the environment

        changed_state = True
        if action_value == 'pickup':
            # print("a")
            changed_idxs = [self.agent_pos.tuple()]
            if np.add.reduce(self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,:3])==0:
                # print('nothing to pick up')
                changed_state = False
            elif np.add.reduce(self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,9:])!=0:
                # print('already holding something')
                changed_state = False
            else:
                # print('picked up', CraftingWorldEnv.translate_state_code(obj_code))
                # old_obs_one_hot = self.obs_one_hot.copy()
                self.obs_one_hot[self.agent_pos.row, self.agent_pos.col,9:] = self.obs_one_hot[self.agent_pos.row, self.agent_pos.col,:3]
                self.obs_one_hot[self.agent_pos.row, self.agent_pos.col,:3] = 0

        elif action_value == 'drop':
            # print("b")
            changed_idxs = [self.agent_pos.tuple()]
            if np.add.reduce(self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,9:])==0:
                changed_state = False  # nothing to drop
            elif np.add.reduce(self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,:8])!=0:
                changed_state = False  # print('can only drop items on an empty spot')
            else:
                # print('dropped', CraftingWorldEnv.translate_state_code(holding_code+1))
                # old_obs_one_hot = self.obs_one_hot.copy()
                self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, :3] = self.obs_one_hot[self.agent_pos.row,
                                                                               self.agent_pos.col, 9:]
                self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, 9:] = 0

        else:
            # print("c")
            changed_state, old_contents_new_loc, changed_idxs = self.__move_agent(action_value)
            self.eval_task_edit(old_contents_new_loc)

        if changed_state == True:
            # test_eval = self.eval_tasks()
            # if not np.array_equal(self.achieved_goal_vector,test_eval):
            #     print("test",test_eval,self.achieved_goal_vector)
            # only need to re-evaluate task success and re-render state if the state has changed
            # self.achieved_goal_vector = self.eval_tasks()
            self.observation_vector = {'observation': self.obs_one_hot, 'desired_goal': self.desired_goal_vector,
                                       'achieved_goal': self.achieved_goal_vector,
                                       'init_observation': self.INIT_OBS_VECTOR}
            # print(changed_idxs)
            self.render_edit(changed_idxs)
            self.observation = {'observation': self.obs_image, 'desired_goal': self.desired_goal,
                                'achieved_goal': self.obs_image, 'init_observation': self.INIT_OBS}
            reward = self.compute_reward(self.achieved_goal_vector[0], self.desired_goal_vector[0], None)
        else:
            reward = -1

        observation = self.observation

        done = True if self.step_num >= self.MAX_STEPS or reward == self.MAX_STEPS else False

        # render if required
        if self.store_gif is True:
            if type(action_value) == Coord:
                self.__render_gif(state_image=self.obs_image, action_label=action_value.name, reward=reward)
            else:
                self.__render_gif(state_image=self.obs_image, action_label=action_value, reward=reward)

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
            return False, None, [self.agent_pos.tuple()]

        new_pos_encoding = self.obs_one_hot[new_pos.tuple()]

        current_pos_encoding = self.obs_one_hot[self.agent_pos.tuple()]
        cant_move_bool = new_pos_encoding[3] * (1 - current_pos_encoding[11]) + new_pos_encoding[4] * (
                    1 - current_pos_encoding[10])
        if cant_move_bool == 1:
            # print("\ncan't move, either tree or rock w/o appropriate tool", self.step_num)
            return False, None, [self.agent_pos.tuple()]
        # old_obs_one_hot = self.obs_one_hot.copy()
        self.obs_one_hot[new_pos.row, new_pos.col, 8:] = current_pos_encoding[8:]
        self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, 8:] = 0
        old_pos = self.agent_pos.tuple()
        self.agent_pos = new_pos
        old_contents_new_loc = self.obs_one_hot[self.agent_pos.tuple()].copy()
        # print(old_contents_new_loc)
        # new_pos_encoding = self.obs_one_hot[self.agent_pos.tuple()]
        # current_obj = np.unravel_index(np.flatnonzero(self.obs_one_hot[self.agent_pos.row, self.agent_pos.col,:8] == 1),
        #                                self.obs_one_hot[self.agent_pos.row, self.agent_pos.col,:8].shape)[0]
        current_obj = np.where(new_pos_encoding[:8] == 1)[0]
        if len(current_obj) == 0:
            # print("no objects on new square, return")
            return True, None, [old_pos,new_pos.tuple()]
        if current_obj[0] in [1,2,6]:
            # print("on axe,hammer, or house, no changes needed, return")
            return True, old_contents_new_loc, [old_pos,new_pos.tuple()]
        elif current_obj[0] in [3,4,5]:
            # print("moved over bread, tree or rock")
            self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, current_obj[0]] = 0
            if current_obj[0] == 4:
                # print("for tree, put sticks there")
                self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, 0] = 1
        elif current_obj[0] == 0:
            # print("moved over sticks:")
            self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,0] *= (1-self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,11])
            self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, 6] = self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, 11]
        elif current_obj[0] == 7:
            # print("moved over wheat:")
            self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, 7] *= (
                        1 - self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, 10])
            self.obs_one_hot[self.agent_pos.row, self.agent_pos.col, 5] = self.obs_one_hot[
                self.agent_pos.row, self.agent_pos.col, 10]
        # print(type(old_pos),type(self.agent_pos.tuple()))
        return True, old_contents_new_loc, [old_pos,new_pos.tuple()]

    def render(self, state=None, mode='Non', tile_size=4):
        """

        :param mode: 'Non' returns the rbg encoding for use in __render_gif(). 'human' also plots for user.
        :param state: the observation needed to render. if None, will render current observation
        :param tile_size: the number of pixels per cell, default 4
        :return: rgb image encoded as a numpy array
        """
        if state is None:
            state = self.obs_one_hot
            a_x, a_y = self.agent_pos.tuple()
        else:
            state_idxs = np.where(state[:, :, 8] == 1)
            a_x, a_y = state_idxs[0][0], state_idxs[1][0]

        height, width = state.shape[0], state.shape[1]
        # print(np.zeros((height,width)))
        # state = np.concatenate((np.zeros((height,width,1),dtype=int),state),axis=2)
        # print(state.shape)
        # print(state.all(axis=2).nonzero())
        objects, agents, holding = np.split(state, [len(OBJECTS), len(OBJECTS) + 1], axis=2)
        objects_n = np.concatenate((np.zeros((height, width, 1), dtype=int), objects), axis=2)
        holding = np.concatenate((np.zeros((height, width, 1), dtype=int), holding), axis=2)
        # print(objects.shape, agents.squeeze(axis=2).shape,holding.shape,"\n")
        # new_state = np.argmax(state, axis=2)
        new_state_h = np.argmax(holding, axis=2)
        # print(new_state_o)

        # objects_n = np.expand_dims(objects, axis=3)
        # objects_n = np.repeat(objects_n, 3, axis=3)
        # COLORS_N_M = np.expand_dims(COLORS_N_M,axis=0)
        # COLORS_N_M = np.expand_dims(COLORS_N_M, axis=0)

        # print(COLORS_N_M.shape,objects_n.shape)
        # print(objects_n)
        img = np.tensordot(objects_n, COLORS_N_M, axes=1)
        img = np.repeat(img, 4, axis=0)
        img = np.repeat(img, 4, axis=1)
        # print(self.agent_pos)

        # print(a_x*4+1,a_y*4+1)
        img[a_x * 4 + 1:a_x * 4 + 3, a_y * 4 + 1:a_y * 4 + 3, :] = 255
        holding = np.max(new_state_h)
        if holding != 0:
            img[a_x * 4 + 2:a_x * 4 + 3, a_y * 4 + 1:a_y * 4 + 3] = COLORS_N[holding]
        # print(img.shape)
        # if np.max(new_state_h!=0):
        #     # test = np.full_like()
        #     COLORS_N_M = np.asarray(COLORS_N)
        #     # COLORS_N_M = np.expand_dims(COLORS_N_M,axis=0)
        #     # COLORS_N_M = np.expand_dims(COLORS_N_M, axis=0)
        #     print(COLORS_N_M.shape)
        #     print(objects.shape)
        #     objects_n = np.expand_dims(objects,axis=3)
        #     objects_n = np.repeat(objects_n,3,axis=3)
        #     # print(objects_n)
        #     print(objects_n.shape)
        #     print(np.tensordot(objects_n,COLORS_N_M).shape)
        #     print("\n",new_state_o)
        #     # new_tile = np.repeat(new_state_o,4,axis=0)
        #     # new_tile = np.repeat(new_tile,4,axis=1)
        #     # print(new_tile)
        #     print(new_state_a)
        #     print(new_state_h)
        #     print(np.max(new_state_h))
        # print(state.shape, new_state)
        # print(new_state_o)
        # print(agents.shape)
        # print(new_state_h)
        # Compute the total grid size
        # width_px, height_px = width * tile_size, height * tile_size
        # img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        #
        if mode == 'human':
            fig2, ax2 = plt.subplots(1)
            ax2.imshow(img)
            fig2.show()

        return img

    def render_edit(self, change_idxs):
        # changes = self.obs_one_hot - old_obs_one_hot

        # change_idxs = np.transpose(
        #     np.unravel_index(np.flatnonzero(changes != 0), changes.shape))
        # vals_to_empty = list(set([(x, y) for x, y, z in change_idxs]))
        # change_idxs = np.unravel_index(np.flatnonzero(changes != 0), changes.shape)
        # change_idxs = np.where(changes != 0)
        # vals_to_empty = list(set([(x, y) for x, y, z in change_idxs]))
        # print(vals_to_empty)
        # for x, y in vals_to_empty:
        # for i in range(len(change_idxs[0])):
        #     x,y = change_idxs[0][i],change_idxs[1][i]
        # print(change_idxs)
        for x,y in change_idxs:
            # print(coordinate)
            # x,y = coordinate[0],coordinate[1]
            # x, y = change_idxs[0][i], change_idxs[1][i]
            # if i!=0:
            #     print(i,vals_to_empty)
            # self.obs_image[x * 4:x * 4 + 4,
            #                     y*4:y*4+4,
            #                     :] = 0
            # print(self.obs_one_hot.shape)
            # print('a',vals_to_empty[i][0])
            # print('b',self.obs_one_hot[vals_to_empty[i][0],vals_to_empty[i][1],9:].shape)
            # print(self.obs_one_hot[vals_to_empty[i][0],vals_to_empty[i][1],:8])
            # print(COLORS_M.shape)
            new_color = np.dot(self.obs_one_hot[x, y, :8], COLORS_M)
            self.obs_image[x * 4:x * 4 + 4, y * 4:y * 4 + 4] = new_color
            # print(np.dot(self.obs_one_hot[x,y,:8],COLORS_M))
            if (x, y) == self.agent_pos.tuple():
                # print("agent is here!")
                self.obs_image[x * 4 + 1:x * 4 + 3, y * 4 + 1:y * 4 + 3, :] = 255
                holding_color = np.dot(self.obs_one_hot[x, y, 9:], COLORS_H)
                self.obs_image[x * 4 + 2:x * 4 + 3, y * 4 + 1:y * 4 + 3] -= holding_color

        # for i in range(len(vals_to_empty[0])):
        #     print(i,vals_to_empty)
        #     self.obs_image[vals_to_empty[0][i]*4:vals_to_empty[0][i]*4+4,
        #                     vals_to_empty[1][i]*4:vals_to_empty[1][i]*4+4,
        #                     :] = 0

    def __render_gif(self, state_image=None, action_label=None, reward=404):

        img2 = state_image if state_image is not None else self.render(mode='Non')
        # im = plt.imshow(img2, animated=True)
        im = self.ax1.imshow(img2, animated=True)
        im2 = self.ax2.imshow(self.observation['desired_goal'])
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

        self.ims.append([im, im2, ttl, txt])

    def sample_state(self):
        """
        produces a observation with one of each object
        :return obs: a sample observation
        :return agent_position: position of the agent within the observation
        """
        diag = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        state = np.zeros(self.observation_vector_space.spaces['observation'].shape, dtype=int)
        state = np.reshape(state, (-1, 1, 12))
        state[:12, 0, :] = diag
        # print(state.shape)
        perm = np.arange(state.shape[0])
        self.np_random.shuffle(perm)
        state = state[perm]
        state = np.reshape(state, (self.STATE_H, -1, 12))
        # print(state.shape)
        # objects = [_ for _ in range(1, 10)]
        # objects = [self.one_hot(i - 1) for i in objects]
        # grid = objects + [[0 for _ in range(self.observation_vector_space.spaces['observation'].shape[2])]
        #                   for _ in range(STATE_W * STATE_H - len(objects))]
        # self.np_random.shuffle(grid)
        #
        # state = np.asarray(grid, dtype=int).reshape(self.observation_vector_space.spaces['observation'].shape)
        # state_idxs = np.unravel_index(np.flatnonzero(state[:,:,8] == 1), state.shape)
        state_idxs = np.where(state[:, :, 8] == 1)
        agent_position = Coord(state_idxs[0][0],
                               state_idxs[1][0],
                               self.STATE_W - 1, self.STATE_H - 1)

        return state, agent_position

    def eval_task_edit(self,old_contents_new_loc):
        """
        changes to the task success will only occur on the agent's location, so we don't need to iterate through
        the whole state space with eval_task after it has already been processed once
        :return:
        """
        # print(old_contents_new_loc)
        # make_bread, eat_bread, build_house, chop_tree, chop_rock, go_to_house, move_axe, move_hammer, move_sticks
        new_objects = np.nonzero(self.obs_one_hot[self.agent_pos.tuple()])[0]
        old_object = np.nonzero(old_contents_new_loc)[0][0] if old_contents_new_loc is not None else 100

        if old_object == 5:
            # there was bread here, agent has eaten it
            self.achieved_goal_vector[0][1] = 1  # eat_bread
        elif old_object == 3:
            # there was a rock here, agent has destroyed it
            self.achieved_goal_vector[0][4] = 1  # chop_rock
        elif old_object == 4:
            # there was a tree here, agent has turned it into sticks
            self.achieved_goal_vector[0][3] = 1  # chop_tree

        # agent is now on a house location
        self.achieved_goal_vector[0][5] = 1 if new_objects[0] == 6 else 0  # go_to_house

        if new_objects[-1] == 8:
            pass # agent not holding anything, so don't have to check the move_object tasks
        elif new_objects[-1] == 9:
            # agent holding sticks
            initial_contents = np.nonzero(self.INIT_OBS_VECTOR[self.agent_pos.tuple()])[0]
            if len(initial_contents) == 0:  # originally an empty space
                self.achieved_goal_vector[0][8] = 1
            elif initial_contents[0] == 0:
                # this is a stick initial location, so sticks haven't been moved
                self.achieved_goal_vector[0][8]=0
            elif initial_contents[0] == 4 and self.achieved_goal_vector[0][3]==1:
                # this is a tree initial spot, and tree was turned into sticks, so sticks haven't been moved
                self.achieved_goal_vector[0][8]=0
            else:
                self.achieved_goal_vector[0][8]=1
        elif new_objects[-1] == 10:
            # agent holding axe
            if old_object == 7:
                self.achieved_goal_vector[0][0] = 1  # make_bread
            initial_contents = np.nonzero(self.INIT_OBS_VECTOR[self.agent_pos.tuple()])[0]
            if len(initial_contents) == 0:  # originally an empty space
                self.achieved_goal_vector[0][6] = 1
            else:
                self.achieved_goal_vector[0][6] = 0 if initial_contents[0] == 1 else 1
        else:
            # agent holding hammer
            if old_object == 0:
                self.achieved_goal_vector[0][2] = 1  # build_house
            initial_contents = np.nonzero(self.INIT_OBS_VECTOR[self.agent_pos.tuple()])[0]
            if len(initial_contents) == 0:  # originally an empty space
                self.achieved_goal_vector[0][7] = 1
            else:
                self.achieved_goal_vector[0][7] = 0 if initial_contents[0] == 2 else 1
        return
        # if old_contents_new_loc is not None:
        #     old_object = np.nonzero(old_contents_new_loc)[0][0]
        #     if old_object == 5:
        #         # there was bread here, agent has eaten it
        #         self.achieved_goal_vector[0][1]=1
        #         return
        #     elif old_object == 3:
        #         # there was a rock here, agent has destroyed it
        #         self.achieved_goal_vector[0][4] = 1
        #         return
        #     elif old_object == 4:
        #         # there was a tree here, agent has turned it into sticks
        #         self.achieved_goal_vector[0][3] = 1
        #         return
        #     # elif old_object == 6:
        #     #     # agent is now on a house location
        #     #     self.achieved_goal_vector[0][5] = 1
        #     #     return
        #     elif old_object == 0:
        #         if new_objects[0] == 6:
        #             # sticks have been turned into a house
        #
        #
        #     print(old_contents_new_loc,np.nonzero(old_contents_new_loc),old_object,new_objects)
        #
        #     # there was something in this spot on the previous timestep
        #     if np.add.reduce(self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,:8]) == 0:
        #         # there isn't an object here now
        #         print(np.nonzero(self.obs_one_hot[self.agent_pos.tuple()]))
        #         if self.INIT_OBS_VECTOR[self.agent_pos.row,self.agent_pos.col,3]==0:
        #             pass
        # else:
        #     # there wasn't anything here, so the only tasks that could have been completed are the move tasks
        #     if self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,10]==1:  # holding axe
        #         move_axe = not self.INIT_OBS_VECTOR[self.agent_pos.row,self.agent_pos.col,10]
        #         if self.INIT_OBS_VECTOR[self.agent_pos.row,self.agent_pos.col,10]==1:
        #             print("move_axe",move_axe, self.INIT_OBS_VECTOR[self.agent_pos.row,self.agent_pos.col,10])
        #     if self.obs_one_hot[self.agent_pos.row,self.agent_pos.col,11]==1:  # holding hammer
        #         move_hammer = not self.INIT_OBS_VECTOR[self.agent_pos.row,self.agent_pos.col,11]
        #         if self.INIT_OBS_VECTOR[self.agent_pos.row,self.agent_pos.col,11]==1:
        #             print("move_hammer",move_hammer, self.INIT_OBS_VECTOR[self.agent_pos.row,self.agent_pos.col,11])
        #     # if self.obs_one_hot[]



    def short_circuit_check(self, a, b, n):  # this fn is basically just np.array_equals, but so much faster
        L = len(a) // n
        for i in range(n):
            j = i * L
            if not all(a[j:j + L] == b[j:j + L]):
                return False
        if not all(a[j + L:] == b[j + L:]):
            return False
        return True

    def compute_reward_equal(self, achieved_goal=None, desired_goal=None, info=None):
        if self.short_circuit_check(desired_goal,achieved_goal,4):
            return self.MAX_STEPS
        else:
            return -1

    def compute_reward_subset(self, achieved_goal=None, desired_goal=None, info=None):
        if np.max(desired_goal-achieved_goal)==0:
            return self.MAX_STEPS
        else:
            return -1

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
            if self.render_flipping is True:
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
