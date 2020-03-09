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
OBJECT_RATIOS = [1, 1, 1, 5, 1, 15, 1, 1]
OBJECT_PROBS = [x/sum(OBJECT_RATIOS) for x in OBJECT_RATIOS]

COLORS = [(110,69,39),(255,105,180),(100,100,200), (100,100,100), (0,128,0), (205,133,63),  (197,91,97),(240,230,140)]
COLORS_rgba = [(110/255.0,69/255.0,39/255.0,.9),(255/255.0,105/255.0,180/255.0,.9),(100/255.0,100/255.0,200/255.0,.9), (100/255.0,100/255.0,100/255.0,.9), (0/255.0,128/255.0,0/255.0,.9), (205/255.0,133/255.0,63/255.0,.9),  (197/255.0,91/255.0,97/255.0,.9),(240/255.0,230/255.0,140/255.0,.9)]

# TODO: check how multigoal worlds work in AI gym, does this affect use of done var, do we give a task to complete, etc
# TODO: maybe explicitly encode x and y as a feature or NOT convolutions - maybe to rbg encoding also?


class CraftingWorldEnv(gym.Env):
    """Custom Crafting that follows gym interface"""
    multi_goal: bool
    metadata = {'render.modes': ['rgba','human']}

    def __init__(self, size=(10, 10), state_type='one_hot', fixed_init_state=False, multi_goal=False, store_gif=False, max_steps=300):
        super(CraftingWorldEnv, self).__init__()
        self.num_rows, self.num_cols = size

        self.state_type = state_type
        if state_type == 'rgb':
            self.observation_space = spaces.Box(low=0,high=255, shape=(self.num_rows*4, self.num_cols*4, 3), dtype=int)
        else:  # state_type == 'one_hot':
            self.observation_space = spaces.Box(low=0,high=1,shape=(self.num_rows, self.num_cols, len(OBJECTS)+1+len(PICKUPABLE)), dtype=int)

        self.fixed_init_state = fixed_init_state
        if self.fixed_init_state:
            self.state = self.fixed_init_state
            self.agent_pos = coord(int(np.where(np.argmax(self.state, axis=2) == 8)[0]),
                                   int(np.where(np.argmax(self.state, axis=2) == 8)[1]), self.num_rows - 1,
                                   self.num_cols - 1)

        else:
            self.state, self.agent_pos = self.sample_state()

        self.init_state = copy.deepcopy(self.state)

        self.ACTIONS = [coord(-1, 0, name='up'), coord(0, 1, name='right'), coord(1, 0, name='down'),
                        coord(0, -1, name='left'), 'pickup', 'drop']  # removed exit as a choice
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.multi_goal = multi_goal
        if self.multi_goal:
            self.goal = None
        else:
            self.goal = 'EatBread'
        self.goal_list = ['EatBread', 'GoToHouse', 'ChopRock', 'ChopTree', 'BuildHouse', 'MakeBread', 'MoveAxe',
                 'MoveHammer', 'MoveSticks']

        self.store_gif = store_gif
        self.fig, self.ax, self.ims = None, None, None
        self.ep_no = 0
        self.step_num = 0
        self.max_steps = max_steps

    def reset(self, init_from_state=None, goal=None):
        """
              init_from_state: If a dictionary state is passed here, the environment
              will reset to that state. Otherwise, the state is randomly initialized.
              """

        if self.store_gif is True and self.step_num != 0:
            # print('debug_final', len(self.ims))
            anim = animation.ArtistAnimation(self.fig, self.ims, interval=100000, blit=False, repeat_delay=1000)
            anim.save('renders/env{}/episode_{}.gif'.format(self.env_id,self.ep_no), writer=animation.PillowWriter(), dpi=100)

        if init_from_state:
            self.state = init_from_state
            self.agent_pos = coord(int(np.where(self.state == 9)[0]),int(np.where(self.state == 9)[1]), self.num_rows-1,
                                   self.num_cols-1)
        else:
            self.state, self.agent_pos = self.sample_state()

        self.init_state = copy.deepcopy(self.state)

        if self.multi_goal:
            if goal:
                self.goal = goal
            else:
                self.goal = np.random.choice(self.goal_list)

        if self.step_num != 0:
            self.ep_no += 1
        self.step_num = 0
        if self.store_gif:
            self.fig, self.ax = plt.subplots(1)
            self.ims = []
            self.render_gif()

    def step(self, action):
        action_value = self.ACTIONS[action]
        self.step_num += 1
        # Execute one time step within the environment
        current_cell = self.state[self.agent_pos.row, self.agent_pos.col]
        objects_at_current_location = current_cell[:len(OBJECTS)]
        holding = current_cell[len(OBJECTS)+1:]
        if action_value == 'pickup':
            #print('obj', objects_at_current_location)
            if objects_at_current_location.any() == 1:  # check there is something at location
                obj_code = np.argmax(objects_at_current_location)
                if holding.any() == 1:
                    pass  #print('already holding something')
                elif obj_code not in [0, 1, 2]:
                    pass  #print('can\'t pick up this object')
                else:
                    #print('picked up', CraftingWorldEnv.translate_state_code(obj_code))
                    self.state[self.agent_pos.row, self.agent_pos.col] = self.one_hot(agent=True, holding=obj_code)
                    #print(self.state[self.agent_pos.row,self.agent_pos.col])
            else:
                pass  #print('nothing at location')
        elif action_value == 'drop':
            if holding.any() == 1:
                holding_code = np.argmax(holding)
                if objects_at_current_location.any() == 1:
                    pass  # print('can only drop items on an empty spot')
                else:
                    # print(holding_code)
                    # print('dropped', CraftingWorldEnv.translate_state_code(holding_code+1))
                    self.state[self.agent_pos.row, self.agent_pos.col] = self.one_hot(obj=holding_code,agent=True)
                    # print(self.state[self.agent_pos.row, self.agent_pos.col])
            else:
                pass
                #print('nothing to drop')

        else:
            self.move_agent(action_value)

        if self.store_gif is True:  # render if required
            if type(action_value) == coord:
                self.render_gif(action_value.name)
            else:
                self.render_gif(action_value)

        observation = self.state
        reward = self.eval_tasks()
        done = False if self.step_num < self.max_steps else True
        if self.multi_goal is False:
            print(self.step_num)
            if reward == 1:
                print('DONE----------------------', self.step_num)
                done = True

        return observation, reward, done, {}

    def move_agent(self, action):
        new_pos = self.agent_pos + action
        if new_pos == self.agent_pos:  # agent is at an edge coordinate
            print('can\'t move, edge of grid')
            return
        objects_at_new_pos = self.state[new_pos.row,new_pos.col][:len(OBJECTS)]
        objects_at_current_pos = self.state[self.agent_pos.row, self.agent_pos.col][:len(OBJECTS)]
        holding_at_current_pos = self.state[self.agent_pos.row, self.agent_pos.col][len(OBJECTS)+1:]
        # print('move',self.agent_pos.row, self.agent_pos.col,objects_at_new_pos, holding_at_current_pos)
        item_in_new_loc = (np.argmax(objects_at_new_pos) if objects_at_new_pos.any() == 1 else None)
        # print('iteminnewloc',item_in_new_loc)
        what_agent_is_holding = (np.argmax(holding_at_current_pos) if holding_at_current_pos.any() == 1 else None)
        if item_in_new_loc == 3:                                                                  # rock in new position
            if what_agent_is_holding != 2:                                                   # agent doesn't have hammer
                #print('can\'t move, rock in way')
                return
            else:                                                                               # agent does have hammer
                item_in_new_loc = None                                                                         # remove rock

        elif item_in_new_loc == 4:                                                                  # tree in new position
            if what_agent_is_holding != 1:                                                       # agent not holding axe
                #print('can\'t move, tree in way')
                return
            else:                                                                                  # agent does have axe
                item_in_new_loc = 0                                                               # turn tree into sticks

        elif item_in_new_loc == 0:                                                                # sticks in new position
            if what_agent_is_holding == 2:                                                            # agent has hammer
                item_in_new_loc = 6                                                              # turn sticks into house

        elif item_in_new_loc == 7:                                                                # wheat in new position
            if what_agent_is_holding == 1:                                                        # agent has axe
                item_in_new_loc = 5                                                              # turn wheat into bread

        elif item_in_new_loc == 5:                                                                # bread in new position
            print('removed bread')
            item_in_new_loc = None
        # print('type2',type(item_in_new_loc),item_in_new_loc)
        self.state[new_pos.row, new_pos.col] = self.one_hot(obj=item_in_new_loc,agent=True, holding=what_agent_is_holding)
        object_in_old_pos = np.argmax(objects_at_current_pos) if objects_at_current_pos.any() == 1 else None
        # print('type',type(object_in_old_pos))
        self.state[self.agent_pos.row, self.agent_pos.col] = self.one_hot(obj=object_in_old_pos,agent=False, holding=None)
        self.agent_pos = new_pos

    def render(self, mode='Non', state=None, tile_size=4):
        """
            Render this grid at a given scale
            :param r: target renderer object
            :param tile_size: tile size in pixels
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
                objects = np.argmax(cell[:len(OBJECTS)]) if cell[:len(OBJECTS)].any() == 1 else None
                agent = cell[len(OBJECTS)]
                holding = np.argmax(cell[len(OBJECTS)+1:]) if cell[len(OBJECTS)+1:].any() == 1 else None

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

    def render_gif(self, action_label=None):
        img2 = self.render(mode='Non')
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


    def sample_state(self):
        num_objects = np.random.randint(4*(self.num_rows//4)*3, 5*(self.num_rows//5)*4)
        #print('num_objects', num_objects)
        objects = [i+1 for i in np.random.choice(len(OBJECTS), num_objects, p=OBJECT_PROBS)]
        objects.append(9)
        objects2 = [self.one_hot(i - 1) for i in objects]
        #print('objects', objects, objects2)
        grid = objects+[0 for _ in range(self.num_rows*self.num_cols-len(objects))]
        grid2 = objects2+[[0 for _ in range(self.observation_space.shape[2])] for _ in range(self.num_rows*self.num_cols-len(objects))]
        #print('grid',grid, grid2)
        random.shuffle(grid2)
        random.shuffle(grid)
        state2 = np.asarray(grid2, dtype=int).reshape(self.observation_space.shape)
        #print('ghskljdlk',state2, np.where(state2 == self.one_hot(8)))
        state = np.asarray(grid, dtype=int).reshape((self.num_rows,self.num_cols))
        #self.render(state=state2, mode='human')
        # print('where',np.where(np.argmax(state2, axis=2) == 8))
        agent_position = coord(int(np.where(np.argmax(state2, axis=2) == 8)[0]), int(np.where(np.argmax(state2, axis=2) == 8)[1]), self.num_rows-1, self.num_cols-1)
        # print(self.one_hot(8), np.argmax(self.one_hot(8)))
        # agent_position = coord(int(np.where(state == 9)[0]),int(np.where(state == 9)[1]),
        #                        self.num_rows-1, self.num_cols-1)

        return state2, agent_position

    def eval_tasks(self):
        # tasks = ['EatBread', 'GoToHouse', 'ChopRock', 'ChopTree', 'BuildHouse', 'MakeBread', 'MoveAxe',
        # 'MoveHammer', "MoveSticks"]
        task_success = {}
        init_objects = {obj: self.get_objects(code, self.init_state) for code,obj in enumerate(OBJECTS)}

        final_objects = {obj: self.get_objects(code, self.state) for code,obj in enumerate(OBJECTS)}
        #print('final_objects', final_objects)
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

        if self.multi_goal is False:
            #print('mgf')
            print(self.goal)
            print(task_success[self.goal])
            task_success = 1 if task_success[self.goal] is True else 0
            if task_success == 1 and self.step_num==1:
                print('issue:\n', init_objects,'\n', final_objects)
            #print('tasksuccess',task_success)
        #task_list = [task_success[key] for key in self.tasks]
        return task_success

    def get_objects(self, code, state):
        code_variants = [code]
        if code<3:
            code_variants.append(code+9)
        # print(code, self.translate_state_code(code), code_variants)
        locations = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                for c in code_variants:
                    if state[i,j,c] == 1:
                        locations += [[i,j]]
        return locations

    def allow_gif_storage(self, store_gif=True, id=None):
        self.store_gif = store_gif
        if self.store_gif is True:
            if id:
                self.env_id=id
            else:
                self.env_id = random.randint(0, 1000000)
            os.makedirs('renders/env{}'.format(self.env_id), exist_ok=False)
            self.fig, self.ax = plt.subplots(1)
            self.ims = []  # storage of step renderings for gif
            self.render_gif()

    @staticmethod
    def translate_state_code(code):
        return OBJECTS[code]

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
    def translate_state_space(state):
        human_readable_response = []
        for row in state:
            human_readable_row = [(CraftingWorldEnv.translate_state_code(code)).center(30) for code in row]
            human_readable_response.append(human_readable_row)
        return human_readable_response





