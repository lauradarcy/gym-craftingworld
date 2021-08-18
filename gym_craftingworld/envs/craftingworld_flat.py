from gym import spaces
from .craftingworld_ray import CraftingWorldEnvRay
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import random
from gym_craftingworld.envs.coordinates import Coord
import matplotlib.patches as mpatches
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

STATE_W = 8
STATE_H = 8

MAX_STEPS = 100


class CraftingWorldEnvFlat(CraftingWorldEnvRay):
    """Custom Crafting environment that follows the gym interface pattern
    """

    metadata = {'render.modes': ['human', 'Non']}

    def __init__(self, size=(STATE_W,STATE_H), max_steps=MAX_STEPS, store_gif=False, render_flipping=False, task_list=TASK_LIST,
                 selected_tasks=TASK_LIST, number_of_tasks=None, stacking=True, reward_style=None):
        super().__init__(size=size, max_steps=max_steps, store_gif=store_gif, render_flipping=render_flipping, task_list=task_list,
                 selected_tasks=selected_tasks, number_of_tasks=number_of_tasks, stacking=stacking, reward_style=reward_style)
        pixel_w, pixel_h = self.STATE_W * 4, self.STATE_H * 4
        self.observation_space = spaces.Box(low=0, high=255, shape=(pixel_w, pixel_h, 3), dtype=int)

    def reset(self, render_next=False):
        """
        reset the environment
        """
        # save episode as gif
        if self.store_gif is True and self.step_num != 0:
            anim = animation.ArtistAnimation(self.fig, self.ims, interval=100000, blit=False, repeat_delay=1000)
            tasknums = '-'.join([str(i) for i in np.where(self.desired_goal_vector[0] == 1)[0]])
            cpmleted = '-'.join([str(i) for i in np.where(self.achieved_goal_vector[0] == 1)[0]])
            if cpmleted != '' or self.ep_no%30==0:
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
            self.desired_goal_vector[0][self.task_list.index(self.selected_tasks[idx])] = 1

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

        return self.obs_image

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

        observation = self.obs_image

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

