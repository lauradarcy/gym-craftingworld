import gym
from gym import spaces
import numpy as np
import copy
from rendering import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from coordinates import coord
import matplotlib.patches as mpatches

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


ACTIONS = [coord(-1, 0, name='up'), coord(0, 1, name='right'), coord(1, 0, name='down'), coord(0, -1, name='left'), 'pickup', 'drop', 'exit']
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
#OBJECT_PROBS = [0.15, 0.05, 0.15, 0.05, 0.1, 0.3, 0.1, 0.1]
OBJECT_PROBS = [0.25, 0.25, 0.5, 0.00, 0.0, 0.0, 0.0, 0.0]

'''
0-8, item (or empty for 0)

10-18, agent holding nothing on item (or on nothing for 10)
20-28, agent holding sticks on item (or on nothing for 20)
30-38, agent holding axe on item (or on nothing for 30)
40-48, agent holding hammer on item (or on nothing for 40)


'''
# COLORS = [(178,34,34),(255,105,180),(0,128,0),(100,100,100),(100,100,200),  (205,133,63), (153,101,21),  (240,230,140),(240,255,240)]
COLORS = [(110,69,39),(255,105,180),(100,100,200), (100,100,100), (0,128,0), (205,133,63),  (197,91,97),(240,230,140)]


COLORS_rgba = [(110/255.0,69/255.0,39/255.0,.9),(255/255.0,105/255.0,180/255.0,.9),(100/255.0,100/255.0,200/255.0,.9), (100/255.0,100/255.0,100/255.0,.9), (0/255.0,128/255.0,0/255.0,.9), (205/255.0,133/255.0,63/255.0,.9),  (197/255.0,91/255.0,97/255.0,.9),(240/255.0,230/255.0,140/255.0,.9)]

class CraftingWorld(gym.Env):
    """Custom Crafting that follows gym interface"""
    metadata = {'render.modes': ['rgb']}

    def __init__(self, size=[10, 10], store_gif = True, res=3, add_objects=[], visible_agent=True, state_obs=False, few_obj=False,
                 use_exit=False, success_function=None, pretty_renderable=False, fixed_init_state=False):
        super(CraftingWorld, self).__init__()
        self.num_rows, self.num_cols = size
        self.nS = self.num_rows * self.num_cols

        self.step_num = 0
        self.ep_no = 0

        self.store_gif = store_gif
        if self.store_gif is True:
            self.fig, self.ax = plt.subplots(1)
            self.ims = []  # storage of step renderings for gif

        self.divisor = len(OBJECTS) + 1
        '''state: each object type gets an int, agent holding nothing is 10 (for example) agent holding item is 20 
        plus int representing object it is holding '''

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(len(ACTIONS))

        self.done = False
        self.observation_space = spaces.Box(low=0,high=(len(OBJECTS)+1)*(len(PICKUPABLE)+2)-1, shape=(self.num_rows,self.num_cols), dtype=int)

        '''each object has a space on the list, then each place on the list stores the number represents its current 
        location '''

        self.state, self.agent_pos = self.sample_state()
        self.init_state = copy.deepcopy(self.state)

        self.fixed_init_state = fixed_init_state
        if self.fixed_init_state:
            self.FIXED_INIT_STATE = np.ndarray([0,0,0,1,0],[0,9,0,0,1])

    def step(self, action):
        self.step_num+=1
        # Execute one time step within the environment
        if action == 'exit':
            self.done = True
        elif action == 'pickup':
            current_val = self.state[self.agent_pos.row,self.agent_pos.col]
            if current_val // self.divisor != 1:
                print('already holding something')
            elif (current_val % self.divisor) not in [1,2,3]:
                print('can\'t pick up this object')
                # self.render_gif('pickup')
                # return
            else:
                print('picked up', CraftingWorld.translate_state_code(current_val%self.divisor))
                self.state[self.agent_pos.row,self.agent_pos.col] = ((current_val % self.divisor)+1)*self.divisor
                # self.render_gif('pickup')
        elif action == 'drop':
            current_val = self.state[self.agent_pos.row, self.agent_pos.col]
            if current_val // self.divisor == 1:
                print('agent isn\'t currently holding anything')
                # self.render_gif('drop')
                # return
            elif (current_val % self.divisor) != 0:
                print('can only drop items on an empty spot')
                # self.render_gif('drop')
                # return
            else:
                print('dropped ', CraftingWorld.translate_state_code(current_val % self.divisor))
                self.state[self.agent_pos.row, self.agent_pos.col] = self.divisor + (current_val // self.divisor)-1
                # self.render_gif('drop')
        else:
            self.move_agent(action)

        if self.store_gif is True:  # render if required
            if type(action) == coord:
                self.render_gif(action.name)
            else:
                self.render_gif(action)

        observation = self.state
        reward = self.eval_tasks()
        done = self.done

        return observation, reward, done, {}

    def move_agent(self,action):
        # print(self.agent_pos)
        new_pos = self.agent_pos + action
        if new_pos == self.agent_pos:  # agent is at an edge coordinate
            print('can\'t move, edge of grid')
            return
        val_at_new_pos = self.state[new_pos.row,new_pos.col]
        val_at_current_pos = self.state[self.agent_pos.row, self.agent_pos.col]
        # CraftingWorld.translate_state_code(val_at_new_pos)
        # CraftingWorld.translate_state_code(val_at_current_pos)
        # print('values',val_at_new_pos,val_at_current_pos)
        # print(CraftingWorld.translate_state_code(val_at_new_pos))
        # print(CraftingWorld.translate_state_code(val_at_current_pos))
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

        # print(self.state[new_pos.row,new_pos.col], self.state[self.agent_pos.row,self.agent_pos.col])
        # print(val_at_current_pos)
        # self.state[new_pos.row,new_pos.col], self.state[self.agent_pos.row, self.agent_pos.col] = val_at_current_pos,0
        # val_at_new_pos = self.state[new_pos.row, new_pos.col]
        # val_at_current_pos = self.state[self.agent_pos.row, self.agent_pos.col]
        # print(CraftingWorld.translate_state_code(val_at_new_pos),
        #       CraftingWorld.translate_state_code(val_at_current_pos))
        # print(self.agent_pos, new_pos)

    def reset(self, init_from_state=None):
        """
              init_from_state: If a dictionary state is passed here, the environment
              will reset to that state. Otherwise, the state is randomly initialized.
              """
        if init_from_state:
            self.state = init_from_state
            self.agent_pos = coord(int(np.where(self.state == 9)[0]),int(np.where(self.state == 9)[1]), self.num_rows-1, self.num_cols-1)
        else:
            self.state, self.agent_pos = self.sample_state()

        self.ep_no += 1
        self.step_num = 0
        if self.store_gif:
            # self.fig = plt.figure()
            # self.ax = self.fig.add_subplot(111)
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
        print('h',title_str)
        ttl = plt.text(0.5, 1.01, title_str+str(self.step_num), horizontalalignment='center', verticalalignment='bottom',
                       transform=self.ax.transAxes)
        patches = [mpatches.Patch(color=COLORS_rgba[i], label="{l}".format(l=OBJECTS[i])) for i in range(len(COLORS))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        self.ims.append([im, ttl])
        print('len',len(self.ims))
        if action_label == 'exit':  #finish episode
            print('a')
            anim = animation.ArtistAnimation(self.fig, self.ims, interval=100000,blit=False, repeat_delay=1000)

            anim.save('test_render{}.gif'.format(self.ep_no), writer=animation.PillowWriter(), dpi=100)

    def sample_state(self):
        num_objects = np.random.randint(4*(self.num_rows//4)*3, 5*(self.num_rows//5)*4)
        objects = [i+1 for i in np.random.choice(len(OBJECTS), num_objects, OBJECT_PROBS)]
        objects.append(9)
        grid = objects+[0 for _ in range(self.num_rows*self.num_cols-len(objects))]
        random.shuffle(grid)
        state = np.asarray(grid, dtype=int).reshape((self.num_rows,self.num_cols))
        agent_position = coord(int(np.where(state == 9)[0]),int(np.where(state == 9)[1]), self.num_rows-1, self.num_cols-1)
        print(agent_position)
        return state, agent_position

    def eval_tasks(self):
        self.tasks = ['EatBread', 'GoToHouse', 'ChopRock', 'ChopTree', 'BuildHouse', 'MakeBread', 'MoveAxe', 'MoveHammer', "MoveSticks"]
        task_success = {}
        init_objects = {obj: self.get_objects(code+1, self.init_state) for code,obj in enumerate(OBJECTS)}
        final_objects = {obj: self.get_objects(code+1, self.state) for code,obj in enumerate(OBJECTS)}
        print(init_objects)


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
        print(task_success)
        task_list = [task_success[key] for key in self.tasks]
        return np.array(task_list)


    # def check_for_movement(self,code):
    #     print('q', code, CraftingWorld.translate_state_code(code))
    #     locations_i = self.get_objects(code, self.init_state)
    #     locations_f = self.get_objects(code, self.state)
    #     print('h', locations_i, locations_f)
    #     if locations_f!=locations_i:
    #         print('movement')
    #         return True
    #     else:
    #         print('no change')
    #     return False

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
        # print(code, code//divisor, code%divisor)
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
            human_readable_row = [(CraftingWorld.translate_state_code(code)).center(30) for code in row]
            human_readable_response.append(human_readable_row)
        return human_readable_response


Q = CraftingWorld([10,10])
# Q.sample_objects()

#Q.reset()
#V = Q.observation_space.sample()
#print(V)
#a,b = V.shape
#print(a,b)
#V = Q.sample_state()
# img = Q.render()

Y = CraftingWorld.translate_state_space(Q.state)

for row in Y:
    print(row)

# Z = CraftingWorld.translate_state_space([[0,1,9],[2,3,4]])
#
# for row in Z:
#     print(row)
#
# tile_size=4
# subdivs=20
#img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

# Draw the grid lines (top and left edges)
# fill_coords(img, point_in_rect(0.01,0.990,0.01,0.990),(200,200,200))
#make_tile(img,(100,100,200))
#img=render()
# plt.imshow(img)
#ax = plt.gca()


# Q.step(ACTIONS[0])



# Gridlines based on minor ticks
#ax.grid(color='w', linewidth=2)
# plt.show()
#
# img = Q.render()
# plt.imshow(img)
# plt.show()
#
# action = 0
# while action != 9:
#     action = int(input('next action:'))
#     if action == 9:
#         break
#     Q.step(ACTIONS[action])
#     #print(Q.state)
#     img = Q.render()
#     plt.imshow(img)
#     plt.show()
# fig=plt.figure()
# ax = fig.add_subplot(111)
# ims=[]



# for _ in range(200):
#     action = int(input('next action:'))
#     if action == 9:
#         break
#     Q.step(ACTIONS[action])
#     print(Q.state)
#     img = Q.render()
#     im = plt.imshow(img, animated=True)
#     ttl = plt.text(0.5, 1.01, LABELS[action], horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
#     patches = [mpatches.Patch(color=COLORS_rgba[i], label="{l}".format(l=OBJECTS[i])) for i in range(len(COLORS))]
#     # put those patched as legend-handles into the legend
#     plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     #txt = plt.text(3,3,LABELS[action])
#     ims.append([im, ttl])
#
#
# ani = animation.ArtistAnimation(fig,ims,interval=100,blit=False,repeat_delay=1000)
#
# ani.save('test.gif', writer=animation.PillowWriter(), dpi=100)
#
# plt.show()
# for _ in range(5):
#     for _ in range(30):
#         action = random.randint(0,5)
#         Q.step(ACTIONS[action])
#     Q.step('exit')
#     Q.reset()
#     fig2, ax2 = plt.subplots(1)
#     ax2.plot(range(100))
#     fig2.show()
#     a = input('i')
#     plt.close(fig2)
# Q.check_for_movement(5)
# fig2, ax2 = plt.subplots(1)
# img = Q.render()
# ax2.imshow(img)
# fig2.show()
#
# Q.eval_tasks()

fig2, ax2 = plt.subplots(1)
img = Q.render()
ax2.imshow(img)
fig2.show()
for _ in range(3):
    action = ACTIONS[int(input('action'))]
    plt.close(fig2)
    for _ in range(5):
        _, reward, _, _ = Q.step(action)
        print(reward)
        fig2, ax2 = plt.subplots(1)
        img = Q.render()
        ax2.imshow(img)
        fig2.show()
        action = ACTIONS[int(input('action'))]
        plt.close(fig2)
    Q.step('exit')
    Q.reset()


    # print(Q.state)

#TODO: put in reward schema

#
# ani.save('test.gif', writer=animation.PillowWriter(), dpi=100)
# Q.step(ACTIONS[6])
# Q.render_gif(ACTIONS[6])


# def is_movie_trash(movie):
#     if movie.genre == superhero:
#         if movie.hero == 'superman':
#             return False
#         else:
#             return True
#     elif movie.has_sequel:
#         return False
#     elif movie.is_sequel:
#         return True
#     else:
#         return Maybe
#


