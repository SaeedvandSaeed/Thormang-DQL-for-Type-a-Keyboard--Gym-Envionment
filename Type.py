import cv2
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import os
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import Robot_Environment
from time import sleep

env = Robot_Environment.Env()
env.reset(randomness=True)

experiment_Number = 0
frame_counter = 0
Recording_frames = True

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *********************

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# **************************


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # print(h,w)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(32)

        # self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding = 2)
        # self.bn4 = nn.BatchNorm2d(32)

        # nn.MaxPool2d(2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        # print(convw,convh)

        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, linear_input_size)
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(x.size(0), -1))

# **************************


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_location(screen_width, screen_height):
    world_width = env.screen_width
    world_height = env.screen_height
    scale_w = screen_width / world_width
    scale_h = screen_height / world_height
    # Midle of object
    return int(env.solid_objx * scale_w), int(env.solid_objy * scale_h), scale_w, scale_h


def get_screen():
    global frame_counter

    screen = env.render1(mode='rgb_array')  # .transpose((2, 0, 1))
    screen_height, screen_width, _ = screen.shape
    # print('Original:', screen.shape)

    obj_location_x, obj_location_y, scale_w, scale_h = get_location(
        screen_width, screen_height)
    # print('OL:', obj_location_x, obj_location_y)

    x_start = (obj_location_x - int(env.objwidth / scale_w * 1.9))
    x_end = (obj_location_x + int(env.objwidth / scale_w * 1.9))
    y_start = ((screen_height - obj_location_y -
                int(env.objheight / scale_h * 1.7)))
    y_end = ((screen_height - obj_location_y +
              int(env.objheight / scale_h * 1.7)))
    screen = screen[y_start:y_end, x_start:x_end]

    # print(type(screen))

    screen = np.asarray(T.functional.to_grayscale(
        T.ToPILImage(mode=None)(screen), num_output_channels=1))

    if(Recording_frames == True):
        frame_counter = frame_counter + 1
        cv2.imwrite("frames/" + str(frame_counter) + ".jpg", screen)

    screen = screen.transpose((1, 0))

    # print('Resides:', screen.shape)

    screen = cv2.Sobel(screen, cv2.CV_64F, 1, 1, ksize=1)

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # ------------ Croped Environment -----------
    # plt.figure(1)
    # plt.imshow(screen.cpu().squeeze(0).permute(1, 0).numpy(), cmap='gray')
    # plt.pause(0.001)

    return resize(screen).unsqueeze(0).to(device)

# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy())
# plt.title('Vision')
# plt.show()

# **************************


BATCH_SIZE = 128
GAMMA = 0.99
LEARNINGRATE = 0.7
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10  # 60000 #$$
TARGET_UPDATE = 100000  # 100 #$$
MODEL_SAVE = 100
num_episodes = 2000
loading_checkpoint = False

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = 3  # @@@@@@@@@@

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0
eps_threshold = 0
eps_threshold_list = []
episode_results = []
CheckpointPath = os.getcwd() + '/Network/' + 'target_net.ckpt'

# ------------------Load Data -------------------
if(loading_checkpoint):
    checkpoint = torch.load(CheckpointPath)
    target_net.load_state_dict(checkpoint['target_model_state_dict'])
    policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']  # i_episode (not used)
    loss = checkpoint['loss']  # reward1 (not used)
    experiment_Number = checkpoint['experiment_Number'] + 1

# **************************

# zero=0
# one=0
# two=0


def actoins_map(action_number):
    if(action_number == 0):  # stop
        return {'PathPattern': 0, 'LeftHandAct': 0, 'RightHandAct': 0}
    elif(action_number == 1):  # right
        return {'PathPattern': 0, 'LeftHandAct': 0, 'RightHandAct': 2}
    elif(action_number == 2):  # left
        return {'PathPattern': 0, 'LeftHandAct': 2, 'RightHandAct': 0}
    # elif(action_number == 3):
    #     return {'PathPattern': 0, 'LeftHandAct': 2, 'RightHandAct': 0}
    # elif(action_number == 4):
    #     return {'PathPattern': 0, 'LeftHandAct': 0, 'RightHandAct': 2}


# ************select_action**************
selectionl = []
selectionr = []
zero_prob = 0.2
one_prob = 0.6
two_prob = 0.3

for i in range(100):
    if(i <= zero_prob * 100):
        selectionl.append(0)
        selectionr.append(0)
    elif(i <= one_prob * 100):
        selectionl.append(1)
        selectionr.append(2)
    elif(i <= two_prob * 100):
        selectionl.append(2)
        selectionr.append(1)


def select_action(state):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # return actoins_map((4))
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            # random_action = random.randrange(n_actions)
            # return torch.tensor([[random_action]], device=device, dtype=torch.long), action_dic

            best_action = policy_net(state).max(1)[1].view(1, 1)
            action_number = best_action.cpu().numpy()[0][0]
            action_dic = actoins_map(action_number)
            # print(action_dic)

            return best_action, action_dic
    else:  # Random
        # return {'PathPattern': 0, 'LeftHandAct': 2, 'RightHandAct': 1}
        # return actoins_map(random.randrange(n_actions))

        global selectionl
        global selectionr

        if (env.ori_object >= 0):   # Guided random selection for bosting the learning time
            random_action = random.choice(selectionl)
        else:
            random_action = random.choice(selectionr)

        # global zero
        # global one
        # global two
        # if(random_action == 0):
        #     zero=zero + 1
        # elif(random_action == 1):
        #     one=one + 1
        # elif(random_action == 2):
        #     two=two + 1
        # print(zero, one, two)

        # random_action = random.randrange(n_actions) # Compleately random
        action_dic = actoins_map(random_action)

        # print(random_action)
        # print()
        # print(env.ori_object)
        return torch.tensor([[random_action]], device=device, dtype=torch.long), action_dic
        # return {'PathPattern': 0, 'LeftHandAct': random.randrange(3), 'RightHandAct': random.randrange(3)}


# **************************
#final_error = None

'''
fig, ax1 = plt.subplots()
color1 = 'tab:green'
color2 = 'tab:blue'
color3 = 'tab:red'

ax1.set_xlabel('Episode')
ax1.set_ylabel('Error', color=color1)
ax2 = ax1.twinx() 
ax2.set_ylabel('Eps_threshold', color=color2)  # we already handled the x-label with ax1
#plt.ylim(ymax = 1, ymin = 0)
ax1.tick_params(axis='y', labelcolor=color1)
ax2.tick_params(axis='y', labelcolor=color2)

def plot_results():
    return
    global final_error
    final_error = torch.tensor(episode_results, dtype=torch.float)
    eps_threshold_t = torch.tensor(eps_threshold_list, dtype=torch.float)
   
    ax1.plot(final_error.numpy(), color=color1)

    ax2.plot(eps_threshold_t.numpy(), color=color2)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    # return  # xxxxxxxxxxxx
    # plt.figure(2)
    # plt.clf()
    # final_error = torch.tensor(episode_results, dtype=torch.float)
    # plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('Error')
    # plt.plot(final_error.numpy())
 
    if len(final_error) >= 1 and len(final_error) < 100:
        means = final_error.unfold(0, len(final_error), 1).mean(1).view(-1)
        #means = torch.cat((torch.zeros(len(final_error) - 1), means))
        # plt.plot(means.numpy())
        ax1.plot(means.numpy(), color=color3)

    if len(final_error) >= 10:
        means = final_error.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        # plt.plot(means.numpy())
        ax1.plot(means.numpy(), color=color3)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        #display.display(plt.gcf())
'''
# ****************************


def optimize_model():
    if len(memory) < BATCH_SIZE:
        # print(memory.sample(1))
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch
    # This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = \
        LEARNINGRATE * ((next_state_values * GAMMA) + reward_batch)

    # print(reward_batch)
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# *****************************


i_episode = 0
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset(randomness=True)
    # time.sleep(5)

    last_screen = get_screen()
    current_screen = last_screen  # get_screen()
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        actions_robot, action_dic = select_action(state)
        state_object, reward1, done1, _ = env.step(action_dic, False)

        print('Rw', reward1)
        reward1 = torch.tensor([(reward1)], device=device)

        #print('action', actions_robot)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done1:
            next_state = current_screen - last_screen
        else:
            next_state = None
        # actions_robot
        # torch.tensor(actions_robot, device=device, dtype=torch.long)
        action = actions_robot.clone().detach()

        # batch_action.long()
        # a.()

        # Store the transition in memory
        memory.push(state, action, next_state, reward1.float())

        # Move to the next state
        state = next_state
        # print(state)
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done1:
            episode_results.append(abs(env.ori_object / 100))  # (t + 1)
            eps_threshold_list.append(eps_threshold)
            # print('eps_threshold',eps_threshold_list)
            # plot_results()
            break
    # Update the target network, copying all weights and biases in DQN

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # if i_episode != 0 and i_episode % MODEL_SAVE == 0:
    #     plt.savefig(os.getcwd() + '/Results/' + 'experiment' +
    #                 str(experiment_Number) + '.eps', format='eps')
    #     np.save(os.getcwd() + '/Results/Output' +
    #             str(experiment_Number) + '.npy', final_error)
        # d = np.load('test3.npy') use it for loading data
        torch.save({
            'target_model_state_dict': target_net.state_dict(),
            'policy_model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': i_episode,
            'loss': reward1,
            'experiment_Number': experiment_Number
        }, CheckpointPath)

plt.savefig(os.getcwd() + '/Results/' + 'experiment' +
            str(experiment_Number) + '.eps', format='eps')
print('Complete.')
env.render1()
env.close()
plt.ioff()
plt.show()
