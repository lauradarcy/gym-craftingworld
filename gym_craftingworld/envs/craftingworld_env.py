import gym
from gym import spaces
import copy
from gym_craftingworld.envs.rendering import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from gym_craftingworld.envs.coordinates import coord
import matplotlib.patches as mpatches
import os

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

PICKUPABLE = [ 'sticks','axe','hammer']
OBJECTS = ['sticks','axe','hammer', 'rock',  'tree',  'bread',  'house', 'wheat']

OBJECT_RATIOS = [1, 1, 1, 1, 1, 1, 1, 1]
OBJECT_PROBS = [x/sum(OBJECT_RATIOS) for x in OBJECT_RATIOS]

COLORS = [(110,69,39),(255,105,180),(100,100,200), (100,100,100), (0,128,0), (205,133,63),  (197,91,97),(240,230,140)]
COLORS_rgba = [(110/255.0,69/255.0,39/255.0,.9),(255/255.0,105/255.0,180/255.0,.9),(100/255.0,100/255.0,200/255.0,.9), (100/255.0,100/255.0,100/255.0,.9), (0/255.0,128/255.0,0/255.0,.9), (205/255.0,133/255.0,63/255.0,.9),  (197/255.0,91/255.0,97/255.0,.9),(240/255.0,230/255.0,140/255.0,.9)]

# TODO: check how multigoal worlds work in AI gym, does this affect use of done var, do we give a task to complete, etc
# TODO: maybe explicitly encode x and y as a feature or NOT convolutions - maybe to rbg encoding also?


class CraftingWorldEnv(gym.Env):
    """Custom Crafting that follows gym interface"""
    metadata = {'render.modes': ['human', 'Non']}

    def __init__(self, size=(10, 10), fixed_init_state=None, store_gif=False, max_steps=300, object_ratios = (1, 1, 1, 1, 1, 1, 1, 1)):
        """
        initialise the environment, change the following args to create a custom environment
        :param size: size of the grid world
        :param fixed_init_state: an initial state to reset to if desired
        :param store_gif: whether or not to store every episode as a gif in a /renders/ subdirectory
        :param max_steps: max number of steps the agent can take
        """
        super(CraftingWorldEnv, self).__init__()
        self.num_rows, self.num_cols = size

        self.max_steps = max_steps

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.num_rows, self.num_cols, len(OBJECTS)+1+len(PICKUPABLE)),
                                            dtype=int)

        self.fixed_init_state = fixed_init_state

        if self.fixed_init_state:
            self.state = self.fixed_init_state
            self.agent_pos = coord(int(np.where(np.argmax(self.state, axis=2) == 8)[0]),
                                   int(np.where(np.argmax(self.state, axis=2) == 8)[1]), self.num_rows - 1,
                                   self.num_cols - 1)
        else:
            self.object_probabilities = [x/sum(object_ratios) for x in object_ratios]
            self.state, self.agent_pos = self.sample_state()

        self.init_state = copy.deepcopy(self.state)

        self.ACTIONS = [coord(-1, 0, name='up'), coord(0, 1, name='right'), coord(1, 0, name='down'),
                        coord(0, -1, name='left'), 'pickup', 'drop']
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.store_gif = store_gif
        self.env_id = None
        self.fig, self.ax, self.ims = None, None, None
        self.ep_no = 0
        self.step_num = 0

    def reset(self):
        """
        reset the environment
        """
        if self.store_gif is True and self.step_num != 0:
            # print('debug_final', len(self.ims))
            anim = animation.ArtistAnimation(self.fig, self.ims, interval=100000, blit=False, repeat_delay=1000)
            anim.save('renders/env{}/episode_{}.gif'.format(self.env_id, self.ep_no), writer=animation.PillowWriter(), dpi=100)

        if self.fixed_init_state:
            self.state = self.fixed_init_state
            self.agent_pos = coord(int(np.where(self.state == 9)[0]),int(np.where(self.state == 9)[1]), self.num_rows-1,
                                   self.num_cols-1)
        else:
            self.state, self.agent_pos = self.sample_state()

        self.init_state = copy.deepcopy(self.state)

        if self.step_num != 0:  # don't increment episode number if resetting after init
            self.ep_no += 1

        self.step_num = 0

        if self.store_gif:
            self.fig, self.ax = plt.subplots(1)
            self.ims = []
            self.__render_gif()

    def step(self, action):
        """
        take a step within the environment
        :param action: integer value within the action_space range
        :return: observations, reward, done, debugging info
        """
        action_value = self.ACTIONS[action]
        self.step_num += 1

        # pull information from agent's current location
        current_cell = self.state[self.agent_pos.row, self.agent_pos.col]
        object_at_current_pos, _, what_agent_is_holding = CraftingWorldEnv.translate_one_hot(current_cell)

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
                    self.state[self.agent_pos.row, self.agent_pos.col] = self.one_hot(agent=True,
                                                                                      holding=object_at_current_pos)

        elif action_value == 'drop':
            if what_agent_is_holding is None:
                pass  # nothing to drop
            else:
                if object_at_current_pos is not None:
                    pass  # print('can only drop items on an empty spot')
                else:
                    # print('dropped', CraftingWorldEnv.translate_state_code(holding_code+1))
                    self.state[self.agent_pos.row, self.agent_pos.col] = self.one_hot(obj=what_agent_is_holding,
                                                                                      agent=True)
        else:
            self.__move_agent(action_value)

        # render if required
        if self.store_gif is True:
            if type(action_value) == coord:
                self.__render_gif(action_value.name)
            else:
                self.__render_gif(action_value)

        observation = self.state
        reward = self.eval_tasks()
        done = False if self.step_num < self.max_steps else True

        return observation, reward, done, {}

    def __move_agent(self, action):
        """
        updates the encoding of two locations in self.state, the old position and the new position.

        first the function adds the coordinates together to get the new location,
        then we pull the contents of each location, including what the agent is holding

        the function performs a series of checks for special cases, i.e. what to do if moving onto a tree location

        then the function updates the encoding of the state

        :param action: one of the movement actions, stored as a coordinate object. coordinate class makes it easier to ensure agent doesn't move outside the grid
        """

        new_pos = self.agent_pos + action

        if new_pos == self.agent_pos:  # agent is at an edge coordinate, so can't move in that direction
            return

        new_pos_encoding = self.state[new_pos.row,new_pos.col]
        object_at_new_pos, _, _ = CraftingWorldEnv.translate_one_hot(new_pos_encoding)

        current_pos_encoding = self.state[self.agent_pos.row, self.agent_pos.col]
        object_at_current_pos, _, what_agent_is_holding = CraftingWorldEnv.translate_one_hot(current_pos_encoding)

        if object_at_new_pos == 3:                                                                  # rock in new position
            if what_agent_is_holding != 2:                                                   # agent doesn't have hammer
                # print('can\'t move, rock in way')
                return
            else:                                                                               # agent does have hammer
                object_at_new_pos = None                                                                     # remove rock

        elif object_at_new_pos == 4:                                                                # tree in new position
            if what_agent_is_holding != 1:                                                       # agent not holding axe
                # print('can\'t move, tree in way')
                return
            else:                                                                                  # agent does have axe
                object_at_new_pos = 0                                                              # turn tree into sticks

        elif object_at_new_pos == 0:                                                             # sticks in new position
            if what_agent_is_holding == 2:                                                            # agent has hammer
                object_at_new_pos = 6                                                             # turn sticks into house

        elif object_at_new_pos == 7:                                                               # wheat in new position
            if what_agent_is_holding == 1:                                                        # agent has axe
                object_at_new_pos = 5                                                              # turn wheat into bread

        elif object_at_new_pos == 5:                                                               # bread in new position
            # print('removed bread')
            object_at_new_pos = None

        # update contents of new position
        self.state[new_pos.row, new_pos.col] = self.one_hot(obj=object_at_new_pos,
                                                            agent=True, holding=what_agent_is_holding)

        # update contents of old position
        self.state[self.agent_pos.row, self.agent_pos.col] = self.one_hot(obj=object_at_current_pos,agent=False, holding=None)

        # update agent's location
        self.agent_pos = new_pos

    def render(self, mode='Non', state=None, tile_size=4):
        """

        :param mode: 'Non' returns the rbg encoding for use in __render_gif(). 'human' also plots for user.
        :param state: the state needed to render. if None, will render current state
        :param tile_size: the number of pixels per cell, default 4
        :return: rgb image encoded as a numpy array
        """
        if state is None:
            state = self.state

        height, width = state.shape[0], state.shape[1]
        # Compute the total grid size
        width_px, height_px = width * tile_size, height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, height):
            for i in range(0, width):
                color = (0,0,0)
                agent_color = None
                agent_holding = None

                cell = state[j, i]
                #objects = np.argmax(cell[:len(OBJECTS)]) if cell[:len(OBJECTS)].any() == 1 else None
                objects, agent, holding = CraftingWorldEnv.translate_one_hot(cell)

                #holding = np.argmax(cell[len(OBJECTS)+1:]) if cell[len(OBJECTS)+1:].any() == 1 else None

                if objects is not None:
                    color = COLORS[objects]
                if agent:
                    agent_color = (250, 250, 250)
                if holding is not None:
                    agent_holding = COLORS[holding]

                tile_img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
                make_tile(tile_img, color, agent_color, agent_holding)

                ymin, ymax = j * tile_size, (j + 1) * tile_size
                xmin, xmax = i * tile_size, (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        if mode == 'human':
            fig2, ax2 = plt.subplots(1)
            ax2.imshow(img)
            fig2.show()

        return img

    def __render_gif(self, action_label=None):
        img2 = self.render(mode='Non')
        im = plt.imshow(img2, animated=True)
        if action_label is None:
            title_str = ''
        else:
            title_str = action_label

        ttl = plt.text(0.5, 1.01, title_str+' '+str(self.step_num), horizontalalignment='center',
                       verticalalignment='bottom', transform=self.ax.transAxes)
        patches = [mpatches.Patch(color=COLORS_rgba[i], label="{l}".format(l=OBJECTS[i])) for i in range(len(COLORS))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        self.ims.append([im, ttl])

    def sample_state(self):
        """
        produces a sample state to start a new episode with
        :return: a random state to start, along with position of agent
        """
        available_tiles = self.num_rows*self.num_cols
        min_num_objects = 3 * max(available_tiles // 20, 1)
        max_num_objects = min(9 * max(available_tiles // 20, 1), available_tiles-1)
        num_objects = np.random.randint(min_num_objects,max_num_objects)

        objects = [i+1 for i in np.random.choice(len(OBJECTS), num_objects, p=self.object_probabilities)]
        objects.append(9)
        objects = [self.one_hot(i - 1) for i in objects]
        grid = objects+[[0 for _ in range(self.observation_space.shape[2])]
                        for _ in range(self.num_rows*self.num_cols-len(objects))]

        random.shuffle(grid)

        state = np.asarray(grid, dtype=int).reshape(self.observation_space.shape)

        agent_position = coord(int(np.where(np.argmax(state, axis=2) == 8)[0]),
                               int(np.where(np.argmax(state, axis=2) == 8)[1]),
                               self.num_rows-1, self.num_cols-1)

        return state, agent_position

    def eval_tasks(self):
        task_success = {}
        init_objects = {obj: self.get_objects(code, self.init_state) for code,obj in enumerate(OBJECTS)}
        final_objects = {obj: self.get_objects(code, self.state) for code,obj in enumerate(OBJECTS)}

        task_success['MakeBread'] = len(final_objects['wheat']) < len(init_objects['wheat'])
        task_success['EatBread'] = (len(final_objects['bread']) + len(final_objects['wheat'])) < (
            len(init_objects['bread']) + len(init_objects['wheat']))
        task_success['BuildHouse'] = len(final_objects['house']) > len(init_objects['house'])
        task_success['ChopTree'] = len(final_objects['tree']) < len(init_objects['tree'])
        task_success['ChopRock'] = len(final_objects['rock']) < len(init_objects['rock'])
        task_success['GoToHouse'] = (self.agent_pos.row, self.agent_pos.col) in final_objects['house']
        task_success['MoveAxe'] = final_objects['axe'] != init_objects['axe']
        task_success['MoveHammer'] = final_objects['hammer'] != init_objects['hammer']
        task_success['MoveSticks'] = False in [stick in init_objects['sticks'] for stick in final_objects['sticks']]

        return task_success

    def get_objects(self, code, state):
        """
        returns the locations for a particular type object within a state

        :param code: the code of the object, which is the index of the object within the one-hot encoding
        :param state: the state to search in
        :return: a list of locations where the object is $[[i_1,j_1],[i_2,j_2],...,[i_n,j_n]]$
        """
        code_variants = [code]
        if code < 3:
            code_variants.append(code+9)
        locations = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                for c in code_variants:
                    if state[i,j,c] == 1:
                        locations += [[i,j]]
        return locations

    def allow_gif_storage(self, store_gif=True):
        """
        turns on or off gif storage, this is a separate function because it create a new subdirectory,
        so wanted the user to have to explicitly call this function

        :param store_gif: a boolean, set to true to turn on gif storage.
        """
        self.store_gif = store_gif
        if self.store_gif is True:
            self.env_id = random.randint(0, 1000000)

            os.makedirs('renders/env{}'.format(self.env_id), exist_ok=False)
            self.fig, self.ax = plt.subplots(1)
            self.ims = []  # storage of step renderings for gif
            self.__render_gif()

    def one_hot(self, obj=None, agent=False, holding=None):
        row = [0 for _ in range(self.observation_space.shape[2])]
        if obj is not None:
            row[obj] = 1
        if agent:
            row[8] = 1
        if holding is not None:
            row[holding+9] = 1
        return row

    @staticmethod
    def translate_one_hot(one_hot_row):
        object_at_location = np.argmax(one_hot_row[:len(OBJECTS)]) if one_hot_row[:len(OBJECTS)].any() == 1 else None
        holding = np.argmax(one_hot_row[len(OBJECTS)+1:]) if one_hot_row[len(OBJECTS)+1:].any() == 1 else None
        agent = one_hot_row[len(OBJECTS)]
        return object_at_location, agent, holding



