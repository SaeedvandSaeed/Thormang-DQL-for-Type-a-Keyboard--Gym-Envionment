import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random


class Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    # ******************************************************

    def __init__(self):
        # ------------------------------
        self.seed()
        self.viewer = None
        self.steps_beyond_done = None
        # ------------------------------
        self.screen_width = 640
        self.screen_height = 480
        self.workSpaceWidth = self.screen_width // 2.5
        self.workSpaceHeight = self.screen_height // 2.5
        self.objwidth = 370.0
        self.objheight = 165.0
        self.armwidth = 160.0 * 1.4
        self.armheight = 210.0 * 1.4
        self.movement_distance = 0
        self.solid_objx = self.screen_width // 2.0 + 50 # MIDDLE 
        self.solid_objy = self.screen_height // 2.0 - self.screen_height // 5 - 10 # MIDDLE 
        self.solid_objx_temp = 0
        self.solid_objx_temp = 0

        self.keyboard_corner_x_offset = 60
        self.hands_location_error = 20

        # Hands desired positions
        self.keys_pos = {'q':[4.5, 7, 40 - 7, 80 - 10], 'p': [22.5, 7, 200 - 25, 80  - 10]} # x(cm), y(cm), x(pxl), y(pxl)

        self.l_arm_pos_x_object = self.solid_objx - self.objwidth // 2 - self.keyboard_corner_x_offset
        self.l_arm_pos_y_object = self.solid_objy - self.objheight + self.keyboard_corner_x_offset // 2
        self.r_arm_pos_x_object = self.solid_objx + self.objwidth // 2 + self.keyboard_corner_x_offset
        self.r_arm_pos_y_object = self.solid_objy - self.objheight + self.keyboard_corner_x_offset // 2

        self.l_desired_pos_x = self.l_arm_pos_x_object + self.keys_pos['q'][2]
        self.l_desired_pos_y = self.l_arm_pos_y_object + self.keys_pos['q'][3]
        self.r_desired_pos_x = self.r_arm_pos_x_object - self.keys_pos['p'][2]
        self.r_desired_pos_y = self.r_arm_pos_y_object + self.keys_pos['p'][3]


        self.ori_object = -1

        self.LeftArmSpeed = 0
        self.RightArmSpeed = 0
        self.PathPattern = 0
        self.rotation_step = 0.01
        self.rotation_step_resolution = 100
        self.rotation_limitation = 25 * self.rotation_step_resolution  # degree * rotation_step resolution
        self.l_arm_movementPath = None
        self.r_arm_movementPath = None
        self.l_step_count = 0
        self.r_step_count = 0
        self.prevous_step_orientation = 0

        high1 = np.array([
            self.l_arm_pos_x_object,
            self.l_arm_pos_y_object,
            self.r_arm_pos_x_object,
            self.r_arm_pos_y_object,
            self.ori_object])

        self.action_space_robot = spaces.MultiDiscrete(
            [1, 2, 3, 4])   # @@@@@@@
        self.observation_space_robot = spaces.Box(
            -high1, high1, dtype=np.float32)

        self.viewer = None
        self.state_obj = None
        self.alfa = 0.9  # Orientation Importance
        self.beta = 0.1  # Porision Importance

        self.steps_beyond_done_obj = None


    # ******************************************************

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # ******************************************************

    def step(self, action, fail):
        step_size = 0.5 * self.rotation_step_resolution
        # --action space: {'PathPattern': 1 , 'LeftHandAct':0,'RightHandAct':0}--
        if(action['PathPattern'] < 0 or action['PathPattern'] > 3) or \
            (action['LeftHandAct'] < 0 or action['LeftHandAct'] > 2) or \
                (action['RightHandAct'] < 0 or action['RightHandAct'] > 2):
            logger.error('Action is not defined!')
            return

        self.PathPattern = action['PathPattern']

        # if action['LeftHandX'] == 0:
        # if action['LeftHandY'] == 0:
        # if action['LeftHandZ'] == 0:

        # if action['RightHandX'] == 0:
        # if action['RightHandY'] == 0:
        # if action['RightHandZ'] == 0:

        if action['LeftHandAct'] == 0:
            self.LeftArmSpeed = 0
        elif action['LeftHandAct'] == 1:
            self.LeftArmSpeed = self.LeftArmSpeed - step_size
        elif action['LeftHandAct'] == 2:
            self.LeftArmSpeed = self.LeftArmSpeed + step_size

        if action['RightHandAct'] == 0:
            self.RightArmSpeed = 0
        if action['RightHandAct'] == 1:
            self.RightArmSpeed = self.RightArmSpeed - step_size
        elif action['RightHandAct'] == 2:
            self.RightArmSpeed = self.RightArmSpeed + step_size

        # -----ori_obj, self.pos_x_object, self.pos_y_object = state_obj----
        state_obj = self.state_obj

        rotation_force = (self.LeftArmSpeed - self.RightArmSpeed)

        # Degree limitation for object rotation
        # if(self.ori_object >= -self.rotation_limitation and self.ori_object <= self.rotation_limitation):
        #     self.ori_object = self.ori_object + rotation_force

        self.state_obj = (self.ori_object, self.l_arm_pos_x_object, self.l_arm_pos_y_object,
            self.r_arm_pos_x_object, self.r_arm_pos_y_object)
        

        location_l_arm = math.sqrt(math.pow(self.l_arm_pos_x_object - self.l_desired_pos_x, 2) +
                             math.pow(self.l_arm_pos_y_object - self.l_desired_pos_y, 2)) + 0.000000001

        location_r_arm = math.sqrt(math.pow(self.r_arm_pos_x_object - self.r_desired_pos_x, 2) +
                             math.pow(self.r_arm_pos_y_object - self.r_desired_pos_y, 2)) + 0.000000001

        done = False
        if(location_l_arm <= 1 and location_r_arm <= 1):
            done = True

        self.l_step_count = len(self.l_arm_movementPath) - 1
        self.r_step_count = len(self.r_arm_movementPath) - 1

        # ---------- For continues movements of tping --------------
        # if(self.l_step_count < len(self.l_arm_movementPath) - 1):
        #     self.l_step_count = self.l_step_count + 1
        # if(self.r_step_count < len(self.r_arm_movementPath) - 1):
        #     self.r_step_count = self.r_step_count + 1

        reward_obj = 0

        if fail:
            # Keyboard just fell! or a key just pressed
            reward_obj = -1
        elif not done:
                reward_obj = 0 
        elif self.steps_beyond_done_obj is None:
            self.steps_beyond_done_obj = 0
            
   
            reward_obj = 1.0 / location_r_arm  / 2 + 1.0 / location_l_arm / 2

            # if (location_l_arm <= 1) or (location_r_arm <= 1):
            #     reward_obj = 1
            # elif self.prevous_step_orientation == self.ori_object:
            #     reward_obj = 1.0 / self.location_r_arm + 1.0 / self.location_l_arm
            # else:
            #     reward_obj = 1.0 / self.location_r_arm + 1.0 / self.location_l_arm
        else:
            if self.steps_beyond_done_obj == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done_obj += 1
        self.prevous_step_orientation = self.ori_object
        return np.array(self.state_obj), (reward_obj), done, {}

    # ******************************************************

    def reset(self, randomness=False):
        self.steps_beyond_done_obj = None
        self.l_step_count = 0
        self.r_step_count = 0

        if(randomness):
            self.ori_object = random.uniform(-2, 2)
        else:
            self.ori_object = 0 

        self.state_obj = (self.ori_object, self.l_arm_pos_x_object, self.l_arm_pos_y_object,
            self.r_arm_pos_x_object, self.r_arm_pos_y_object,)

        self.RightArmSpeed = 0
        self.LeftArmSpeed = 0
        self.prevous_step_orientation = 0

        if(randomness):
            self.l_arm_pos_x_object = random.uniform(
                self.solid_objx - self.objwidth // 2 - self.keyboard_corner_x_offset + self.hands_location_error,
                self.solid_objx - self.objwidth // 2 - self.keyboard_corner_x_offset - self.hands_location_error)

            self.l_arm_pos_y_object = random.uniform(
                self.solid_objy - self.objheight - self.hands_location_error,
                self.solid_objy - self.objheight + self.hands_location_error)

            self.r_arm_pos_x_object = random.uniform(
                self.solid_objx + self.objwidth // 2 + self.keyboard_corner_x_offset + self.hands_location_error,
                self.solid_objx + self.objwidth // 2 + self.keyboard_corner_x_offset - self.hands_location_error)

            self.r_arm_pos_y_object = random.uniform(
                self.solid_objy - self.objheight - self.hands_location_error,
                self.solid_objy - self.objheight + self.hands_location_error)

            self.solid_objx_temp = self.solid_objx + random.uniform(-20, 20)
            self.solid_objy_temp = self.solid_objy + random.uniform(-20, 20)

        else:
            self.l_arm_pos_x_object = self.solid_objx - self.objwidth // 2 - self.keyboard_corner_x_offset
            self.l_arm_pos_y_object = self.solid_objy - self.objheight + self.keyboard_corner_x_offset // 2
            self.r_arm_pos_x_object = self.solid_objx + self.objwidth // 2 + self.keyboard_corner_x_offset
            self.r_arm_pos_y_object = self.solid_objy - self.objheight + self.keyboard_corner_x_offset // 2

        #print(self.pos_x_object, self.pos_y_object)

        self.l_arm_movementPath = self.get_line(int(self.l_arm_pos_x_object),  int(self.l_arm_pos_y_object),
                                          int(self.l_desired_pos_x), int(self.l_desired_pos_y))
        
        self.r_arm_movementPath = self.get_line(int(self.r_arm_pos_x_object),  int(self.r_arm_pos_y_object),
                                          int(self.r_desired_pos_x), int(self.r_desired_pos_y))
        self.steps_beyond_done = None
        return #np.array(self.state_obj)

    # ******************************************************
    def get_line(self, x1, y1, x2, y2):
        points = []
        issteep = abs(y2-y1) > abs(x2-x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        rev = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            rev = True
        deltax = x2 - x1
        deltay = abs(y2-y1)
        error = int(deltax / 2)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1

        for q in range(x1, x2 + 1):
            if issteep:
                points.append((y, q))
            else:
                points.append((q, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax

        # Reverse the list if the coordinates were reversed
        if rev:
            points.reverse()

        return points

    # ******************************************************
    def render1(self, mode='human'):
        stepsize = self.screen_width // 10

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(
                self.screen_width, self.screen_height)

            # ------------------------
            # WorkSpace
            l, r, t, b = -self.workSpaceWidth, self.workSpaceWidth, \
                self.workSpaceHeight, - self.workSpaceHeight
            workspace = rendering.make_polygon(
                [(l, b), (l, t), (r, t), (r, b)], False)
            workspace.set_color(.0, .0, .0)
            self.workspacetrans = rendering.Transform(translation=(
                self.screen_width/2.0, self.screen_height/3.0 - 20), rotation=0.00)
            workspace.add_attr(self.workspacetrans)
            self.viewer.add_geom(workspace)
            # ------------------------
            #  Desired Zone
            # l, r, t, b = -self.workSpaceWidth//2, self.workSpaceWidth//2, \
            #     self.workSpaceHeight//2, - self.workSpaceHeight//2
            # axleoffset = self.objheight/4.0
            # zone = rendering.make_polygon(
            #     [(l, b), (l, t), (r, t), (r, b)], False)
            # zone.set_color(.0, 255, 0)
            # self.zonetrans = rendering.Transform(translation=(
            #     self.screen_width/2.0, self.workSpaceHeight + 12), rotation=0.00)
            # zone.add_attr(self.zonetrans)
            # self.viewer.add_geom(zone)
            # ------------------------
            # Desired Zone
            radious = 10
            zone = rendering.make_circle(radious, 30, False)
            zone.set_color(0, 165, 255)
            self.zonetrans = rendering.Transform(translation=(
                self.l_desired_pos_x, self.l_desired_pos_y), rotation=-0.00)
            zone.add_attr(self.zonetrans)
            self.viewer.add_geom(zone)
            # ------------------------
            # Background
            Background = rendering.Image(
                'Environment/Background.jpg', self.screen_width, self.screen_height)
            Background.set_color(255, 255, 255)
            self.Backgroundtrans = rendering.Transform(rotation=-0.00)
            Background.add_attr(self.Backgroundtrans)
            self.viewer.add_geom(Background)
            # ------------------------
            # Keyboard
            obj1 = rendering.Image(
                'Environment/Object_Keyboard.png', self.objwidth, self.objheight)
            obj1.set_color(255, 255, 255)
            self.objtrans = rendering.Transform(rotation=np.radians(self.ori_object))
            obj1.add_attr(self.objtrans)
            self.viewer.add_geom(obj1)
            # ------------------------
            # left Arm
            larm = rendering.Image(
                'Environment/larm.png', self.armwidth, self.armheight)
            larm.set_color(255, 255, 255)
            self.larmtrans = rendering.Transform(rotation=-0.1)
            larm.add_attr(self.larmtrans)
            self.viewer.add_geom(larm)
            # ------------------------
            # right Arm
            rarm = rendering.Image(
                'Environment/rarm.png', self.armwidth, self.armheight)
            rarm.set_color(255, 255, 255)
            self.rarmtrans = rendering.Transform(rotation=0.1)
            rarm.add_attr(self.rarmtrans)
            self.viewer.add_geom(rarm)
            # ------------------------

        if self.state_obj is None:
            return None

        x = self.state_obj
        orientation_state = x[0]
 
        posl = self.l_arm_movementPath[self.l_step_count]
        posr = self.r_arm_movementPath[self.r_step_count]


        self.objtrans.set_translation(self.solid_objx_temp , self.solid_objy_temp)

        self.Backgroundtrans.set_translation(self.screen_width // 2, self.screen_height // 2)

        disR = math.tanh(np.radians((orientation_state / self.rotation_step_resolution))
                         ) * (self.objwidth / 2)
  
        self.larmtrans.set_translation(posl[0], posl[1])
        self.rarmtrans.set_translation(posr[0], posr[1])

        
        self.l_arm_pos_x_object = posl[0]
        self.l_arm_pos_y_object = posl[1]
        self.movement_distance = \
            math.sqrt(math.pow(self.l_arm_pos_x_object - self.l_arm_movementPath[self.l_step_count - 1][0], 2)
                      + math.pow(self.l_arm_pos_y_object - self.l_arm_movementPath[self.l_step_count - 1][1], 2))

        self.r_arm_pos_x_object = posr[0]
        self.r_arm_pos_y_object = posr[1]
        self.movement_distance = \
            math.sqrt(math.pow(self.r_arm_pos_x_object - self.r_arm_movementPath[self.r_step_count - 1][0], 2)
                      + math.pow(self.r_arm_pos_y_object - self.r_arm_movementPath[self.r_step_count - 1][1], 2))


        # print(self.movement_distance)
       # self.objtrans.set_rotation(np.radians(-1))
        self.objtrans.set_rotation(np.radians(self.ori_object))

        # self.larmtrans.set_rotation(np.radians(
        #     orientation_state * self.rotation_step))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # ******************************************************

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
