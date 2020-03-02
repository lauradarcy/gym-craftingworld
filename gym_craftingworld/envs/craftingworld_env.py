import gym
from gym import spaces
import copy
from gym_craftingworld.envs.rendering import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from gym_craftingworld.envs.coordinates import coord
import matplotlib.patches as mpatches

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


#ACTIONS = [coord(-1, 0, name='up'), coord(0, 1, name='right'), coord(1, 0, name='down'), coord(0, -1, name='left'), 'pickup', 'drop', 'exit']
LABELS = ['up','right','down','left','pickup','drop','exit']


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


AGENT = 'agent'
PICKUPABLE = [ 'sticks','axe','hammer']
BLOCKING = ['rock', 'tree']

HOLDING = 'holding'
CLIP = True
OBJECTS = ['sticks','axe','hammer', 'rock',  'tree',  'bread',  'house', 'wheat']
OBJECT_PROBS = [0.25, 0.25, 0.5, 0.00, 0.0, 0.0, 0.0, 0.0]

'''
0-8, item (or empty for 0)

10-18, agent holding nothing on item (or on nothing for 10)
20-28, agent holding sticks on item (or on nothing for 20)
30-38, agent holding axe on item (or on nothing for 30)
40-48, agent holding hammer on item (or on nothing for 40)


'''
COLORS = [(110,69,39),(255,105,180),(100,100,200), (100,100,100), (0,128,0), (205,133,63),  (197,91,97),(240,230,140)]

COLORS_rgba = [(110/255.0,69/255.0,39/255.0,.9),(255/255.0,105/255.0,180/255.0,.9),(100/255.0,100/255.0,200/255.0,.9), (100/255.0,100/255.0,100/255.0,.9), (0/255.0,128/255.0,0/255.0,.9), (205/255.0,133/255.0,63/255.0,.9),  (197/255.0,91/255.0,97/255.0,.9),(240/255.0,230/255.0,140/255.0,.9)]

# TODO: one-hot encoding, so things aren't similar to each other, tensor like image where one hot is z dimension
# TODO: maybe explicitly encode x and y as a feature or NOT convolutions - maybe to rbg encoding also?


class CraftingWorldEnv(gym.Env):
    """Custom Crafting that follows gym interface"""
    metadata = {'render.modes': ['rgb']}

    def __init__(self, size=[10, 10], store_gif=True, fixed_init_state=False):
        super(CraftingWorldEnv, self).__init__()
        self.num_rows, self.num_cols = size
        self.nS = self.num_rows * self.num_cols

        self.step_num = 0
        self.ep_no = 0



        self.divisor = len(OBJECTS) + 1
        '''state: each object type gets an int, agent holding nothing is 10 (for example) agent holding item is 20 
        plus int representing object it is holding '''

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.ACTIONS = [coord(-1, 0, name='up'), coord(0, 1, name='right'), coord(1, 0, name='down'),
                   coord(0, -1, name='left'), 'pickup', 'drop', 'exit']

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.done = False
        self.observation_space = spaces.Box(low=0,high=(len(OBJECTS)+1)*(len(PICKUPABLE)+2)-1, shape=(self.num_rows,self.num_cols), dtype=int)

        '''each object has a space on the list, then each place on the list stores the number represents its current 
        location '''

        self.state, self.agent_pos = self.sample_state()
        self.init_state = copy.deepcopy(self.state)

        self.store_gif = store_gif
        if self.store_gif is True:
            self.fig, self.ax = plt.subplots(1)
            self.ims = []  # storage of step renderings for gif
            self.render_gif()

        self.fixed_init_state = fixed_init_state
        if self.fixed_init_state:
            raise NotImplementedError

    def step(self, action):
        action_value = self.ACTIONS[action]
        self.step_num += 1
        # Execute one time step within the environment
        if action_value == 'exit':
            self.done = True
        elif action_value == 'pickup':
            current_val = self.state[self.agent_pos.row,self.agent_pos.col]
            if current_val // self.divisor != 1:
                print('already holding something')
            elif (current_val % self.divisor) not in [1,2,3]:
                print('can\'t pick up this object')
            else:
                print('picked up', CraftingWorldEnv.translate_state_code(current_val % self.divisor))
                self.state[self.agent_pos.row,self.agent_pos.col] = ((current_val % self.divisor)+1)*self.divisor
        elif action_value == 'drop':
            current_val = self.state[self.agent_pos.row, self.agent_pos.col]
            if current_val // self.divisor == 1:
                print('agent isn\'t currently holding anything')
            elif (current_val % self.divisor) != 0:
                print('can only drop items on an empty spot')
            else:
                print('dropped ', CraftingWorldEnv.translate_state_code(current_val % self.divisor))
                self.state[self.agent_pos.row, self.agent_pos.col] = self.divisor + (current_val // self.divisor)-1
        else:
            self.move_agent(action_value)

        if self.store_gif is True:  # render if required
            if type(action_value) == coord:
                self.render_gif(action_value.name)
            else:
                self.render_gif(action_value)

        observation = self.state
        reward = self.eval_tasks()
        done = self.done

        return observation, reward, done, {}

    def move_agent(self, action):
        new_pos = self.agent_pos + action
        if new_pos == self.agent_pos:  # agent is at an edge coordinate
            print('can\'t move, edge of grid')
            return
        val_at_new_pos = self.state[new_pos.row,new_pos.col]
        val_at_current_pos = self.state[self.agent_pos.row, self.agent_pos.col]

        item_in_new_loc = val_at_new_pos % self.divisor
        what_agent_is_holding = val_at_current_pos // self.divisor
        if item_in_new_loc == 4:                                                                  # rock in new position
            if what_agent_is_holding != 4:                                                   # agent doesn't have hammer
                print('can\'t move, rock in way')
                return
            else:                                                                               # agent does have hammer
                val_at_new_pos = 0                                                                         # remove rock

        if item_in_new_loc == 5:                                                                  # tree in new position
            if what_agent_is_holding != 3:                                                       # agent not holding axe
                print('can\'t move, tree in way')
                return
            else:                                                                                  # agent does have axe
                val_at_new_pos = 1                                                               # turn tree into sticks

        if item_in_new_loc == 1:                                                                # sticks in new position
            if what_agent_is_holding == 4:                                                            # agent has hammer
                val_at_new_pos = 7                                                              # turn sticks into house

        if item_in_new_loc == 8:                                                                # wheat in new position
            if what_agent_is_holding == 3:                                                        # agent has axe
                val_at_new_pos = 6                                                              # turn wheat into bread

        if item_in_new_loc == 6:                                                                # bread in new position
            print('removed bread')
            val_at_new_pos = 0

        self.state[new_pos.row, new_pos.col], self.state[self.agent_pos.row, self.agent_pos.col], self.agent_pos = \
            val_at_new_pos % self.divisor + (val_at_current_pos // self.divisor)*self.divisor, \
            val_at_current_pos % self.divisor, new_pos

    def reset(self, init_from_state=None):
        """
              init_from_state: If a dictionary state is passed here, the environment
              will reset to that state. Otherwise, the state is randomly initialized.
              """
        if init_from_state:
            self.state = init_from_state
            self.agent_pos = coord(int(np.where(self.state == 9)[0]),int(np.where(self.state == 9)[1]), self.num_rows-1,
                                   self.num_cols-1)
        else:
            self.state, self.agent_pos = self.sample_state()

        self.ep_no += 1
        self.step_num = 0
        if self.store_gif:
            self.fig, self.ax = plt.subplots(1)
            self.ims = []

    def render(self, state=None, tile_size=4):
        """
            Render this grid at a given scale
            :param r: target renderer object
            :param tile_size: tile size in pixels
            """
        if state is None:
            state = self.state
        height, width = state.shape
        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        # Render the grid

        for j in range(0, height):
            for i in range(0, width):
                color = (0,0,0)
                agent_color = None
                agent_holding = None
                cell = state[j, i]
                if cell % self.divisor != 0:
                    color = COLORS[cell % self.divisor - 1]
                if cell // self.divisor != 0:
                    agent_color = (250,250,250)
                    if cell//self.divisor > 1:
                        agent_holding = COLORS[cell // self.divisor - 2]
                tile_img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)

                make_tile(tile_img, color, agent_color, agent_holding)

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def render_gif(self, action_label=None):
        img2 = self.render()
        im = plt.imshow(img2, animated=True)
        if action_label is None:
            title_str = ''
        else:
            title_str = action_label

        ttl = plt.text(0.5, 1.01, title_str+str(self.step_num), horizontalalignment='center', verticalalignment='bottom',
                       transform=self.ax.transAxes)
        patches = [mpatches.Patch(color=COLORS_rgba[i], label="{l}".format(l=OBJECTS[i])) for i in range(len(COLORS))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        self.ims.append([im, ttl])

        if action_label == 'exit':  #finish episode
            anim = animation.ArtistAnimation(self.fig, self.ims, interval=100000,blit=False, repeat_delay=1000)
            anim.save('test_render{}.gif'.format(self.ep_no), writer=animation.PillowWriter(), dpi=100)

    def sample_state(self):
        num_objects = np.random.randint(4*(self.num_rows//4)*3, 5*(self.num_rows//5)*4)
        objects = [i+1 for i in np.random.choice(len(OBJECTS), num_objects, OBJECT_PROBS)]
        objects.append(9)
        grid = objects+[0 for _ in range(self.num_rows*self.num_cols-len(objects))]
        random.shuffle(grid)
        state = np.asarray(grid, dtype=int).reshape((self.num_rows,self.num_cols))
        agent_position = coord(int(np.where(state == 9)[0]),int(np.where(state == 9)[1]),
                               self.num_rows-1, self.num_cols-1)
        return state, agent_position

    def eval_tasks(self):
        # tasks = ['EatBread', 'GoToHouse', 'ChopRock', 'ChopTree', 'BuildHouse', 'MakeBread', 'MoveAxe',
        # 'MoveHammer', "MoveSticks"]
        task_success = {}
        init_objects = {obj: self.get_objects(code+1, self.init_state) for code,obj in enumerate(OBJECTS)}
        final_objects = {obj: self.get_objects(code+1, self.state) for code,obj in enumerate(OBJECTS)}

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

        #task_list = [task_success[key] for key in self.tasks]
        return task_success

    def get_objects(self, code, state):
        code_variants = [(i * self.divisor) + code for i in range(5)]
        locations = []
        for c in code_variants:
            rows, cols = np.where(state == c)
            coordinates = list(zip(rows, cols))
            locations += coordinates
        return locations

    @staticmethod
    def translate_state_code(code):
        divisor = len(OBJECTS) + 1

        if code // divisor == 0:
            string_part_one = ''
        elif code // divisor == 1:
            string_part_one = 'Agent'
        else:
            string_part_one = 'Agent holding ' + PICKUPABLE[code // divisor - 2]
        if code % divisor == 0:
            string_part_two = ''
        elif code // divisor != 0:
            string_part_two = ', ' + OBJECTS[code % divisor - 1]
        else:
            string_part_two = OBJECTS[code % divisor - 1]
        return string_part_one + string_part_two

    @staticmethod
    def translate_state_space(state):
        human_readable_response = []
        for row in state:
            human_readable_row = [(CraftingWorldEnv.translate_state_code(code)).center(30) for code in row]
            human_readable_response.append(human_readable_row)
        return human_readable_response





