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
        self.screen_width = 600
        self.screen_height = 400
        self.workSpaceWidth = self.screen_width // 2.5
        self.workSpaceHeight = self.screen_height // 2.5
        self.objwidth = 200.0
        self.objheight = 80.0
        self.armwidth = 50.0 * 1.4
        self.armheight = 200.0 * 1.4
        self.movement_distance = 0

        self.pos_x_object = 0
        self.pos_y_object = 0
        self.ori_object = 0

        self.LeftArmSpeed = 0
        self.RightArmSpeed = 0
        self.PathPattern = 0
        self.rotation_step = 0.01
        self.rotation_step_resolution = 100
        self.rotation_limitation = 25 * self.rotation_step_resolution  # degree * rotation_step resolution
        self.movementPath = None
        self.step_count = 0
        self.prevous_step_orientation = 0

        high1 = np.array([
            self.pos_x_object,
            self.pos_y_object,
            self.ori_object])

        self.action_space_robot = spaces.MultiDiscrete(
            [1, 2, 3, 4])   # @@@@@@@
        self.observation_space_robot = spaces.Box(
            -high1, high1, dtype=np.float32)

        self.viewer = None
        self.state_obj = None
        self.alfa = 0.9  # Orientation Importance
        self.beta = 0.1  # Porision Importance

        # Final desired pos cm
        self.Desired_pos_x = int(self.screen_width // 2.0)
        self.Desired_pos_y = int(self.screen_height //
                                 4.0 - 20)  # Final pos cm

        self.steps_beyond_done_obj = None


    # ******************************************************

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # ******************************************************

    def step1(self, action, fail):
        step_size = 0.5 * self.rotation_step_resolution
        # --action space: {'PathPattern': 1 , 'LeftHandAct':0,'RightHandAct':0}--
        if(action['PathPattern'] < 0 or action['PathPattern'] > 3) or \
            (action['LeftHandAct'] < 0 or action['LeftHandAct'] > 2) or \
                (action['RightHandAct'] < 0 or action['RightHandAct'] > 2):
            logger.error('Action is not defined!')
            return

        self.PathPattern = action['PathPattern']

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
        if(self.ori_object >= -self.rotation_limitation and self.ori_object <= self.rotation_limitation):
            self.ori_object = self.ori_object + rotation_force

        self.state_obj = (self.ori_object, self.pos_x_object,
                          self.pos_y_object)

        location = math.sqrt(math.pow(self.pos_x_object - self.Desired_pos_x, 2) +
                             math.pow(self.pos_y_object - self.Desired_pos_y, 2))

        done = False
        if(location <= 5):
            done = True

        self.step_count = self.step_count + 1
        reward_obj = 0

        if fail:
            # Keyboard just fell! or a key just pressed
            reward_obj = -1
        elif not done:
            if abs(self.ori_object) < 5 * self.rotation_step_resolution:
                reward_obj = 1 
            elif self.prevous_step_orientation == self.ori_object:
                reward_obj = 0#1.0 / abs(self.ori_object)
            else:
                reward_obj = 1.0 / abs(self.ori_object)
                # reward_obj = self.alfa * \
                #     (1 - 1.0 / abs((self.prevous_step_orientation) - (self.ori_object))) + (1.0 / abs(self.ori_object))
        elif self.steps_beyond_done_obj is None:
            self.steps_beyond_done_obj = 0
            if abs(self.ori_object) < 5 * self.rotation_step_resolution:
                reward_obj = 1
            elif self.prevous_step_orientation == self.ori_object:
                reward_obj = 0#1.0 / abs(self.ori_object)
            else:
                reward_obj = 1.0 / abs(self.ori_object)
                # reward_obj = self.alfa * \
                #     (1 - 1.0 / (abs(self.prevous_step_orientation) - abs(self.ori_object))) + (1.0 / abs(self.ori_object))
                # + self.beta * (1.0 / (math.sqrt(math.pow(self.Desired_pos_x - self.pos_x_object, 2) +
                #                              math.pow(self.Desired_pos_y - self.pos_y_object, 2))) + 1)
        else:
            if self.steps_beyond_done_obj == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done_obj += 1
        self.prevous_step_orientation = self.ori_object
        return np.array(self.state_obj), (reward_obj), done, {}

    # ******************************************************

    def reset1(self, randomness=False):
        self.steps_beyond_done_obj = None
        self.step_count = 0

        if(randomness):
            self.ori_object = random.uniform(-self.rotation_limitation,
                                             self.rotation_limitation)
        else:
            self.ori_object = 25 * self.rotation_step_resolution

        self.state_obj = (self.ori_object, self.pos_x_object,
                        self.pos_y_object)

        self.RightArmSpeed = 0
        self.LeftArmSpeed = 0
        self.prevous_step_orientation = 0
        safe_zone = 15

        if(randomness):
            self.pos_x_object = random.uniform(
                int((self.screen_width // 2) -
                    (self.workSpaceWidth // 2)) + safe_zone,
                int((self.workSpaceWidth // 2) + (self.screen_width // 2)) - safe_zone)

            self.pos_y_object = random.uniform(
                int((self.screen_height // 2) -
                    (self.workSpaceHeight // 2) + (self.objheight // 2)) + safe_zone,
                int((self.workSpaceHeight // 2) + (self.screen_height // 2) - (self.objheight // 2)) - safe_zone)
        else:
            self.pos_x_object = self.workSpaceWidth // 2 + 100
            self.pos_y_object = self.workSpaceHeight // 2 + 100

        #print(self.pos_x_object, self.pos_y_object)

        self.movementPath = self.get_line(int(self.pos_x_object),  int(self.pos_y_object),
                                          int(self.Desired_pos_x), int(self.Desired_pos_y))
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
                self.Desired_pos_x, self.Desired_pos_y), rotation=-0.00)
            zone.add_attr(self.zonetrans)
            self.viewer.add_geom(zone)
            # ------------------------
            # left Arm
            larm = rendering.Image(
                'Environment/larm.jpg', self.armwidth, self.armheight)
            larm.set_color(255, 255, 255)
            self.larmtrans = rendering.Transform(rotation=-0.1)
            larm.add_attr(self.larmtrans)
            self.viewer.add_geom(larm)
            # ------------------------
            # right Arm
            rarm = rendering.Image(
                'Environment/rarm.jpg', self.armwidth, self.armheight)
            rarm.set_color(255, 255, 255)
            self.rarmtrans = rendering.Transform(rotation=0.1)
            rarm.add_attr(self.rarmtrans)
            self.viewer.add_geom(rarm)
            # ------------------------
            # Keyboard
            obj1 = rendering.Image(
                'Environment/Object_Keyboard.jpg', self.objwidth, self.objheight)
            obj1.set_color(255, 255, 255)
            self.objtrans = rendering.Transform(rotation=-0.00)
            obj1.add_attr(self.objtrans)
            self.viewer.add_geom(obj1)
            # ------------------------

        if self.state_obj is None:
            return None

        x = self.state_obj
        orientation_state = x[0]
        objx = x[1] * stepsize + self.screen_width // 2.0  # MIDDLE OF CART
        objy = x[2] * stepsize + self.screen_height // 4.0  # MIDDLE OF CART
        pos = self.movementPath[self.step_count]

        self.objtrans.set_translation(pos[0], pos[1])
        disR = math.tanh(np.radians((orientation_state / self.rotation_step_resolution))
                         ) * (self.objwidth / 2)
        self.larmtrans.set_translation(
            pos[0] - self.objwidth // 2 - 20, pos[1] - self.objheight - 20 - disR)
        self.rarmtrans.set_translation(
            pos[0] + self.objwidth // 2 + 20, pos[1] - self.objheight - 20 + disR)

        self.pos_x_object = pos[0]
        self.pos_y_object = pos[1]
        self.movement_distance = \
            math.sqrt(math.pow(self.pos_x_object - self.movementPath[self.step_count - 1][0], 2)
                      + math.pow(self.pos_y_object - self.movementPath[self.step_count - 1][1], 2))
       # print(self.movement_distance)

        self.objtrans.set_rotation(np.radians(
            orientation_state * self.rotation_step))

        # self.larmtrans.set_rotation(np.radians(
        #     orientation_state * self.rotation_step))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # ******************************************************

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
