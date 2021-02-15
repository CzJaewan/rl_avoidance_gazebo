import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
import argparse
from mpi4py import MPI
import copy

from gym import spaces

from torch.optim import Adam
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from model.utils import soft_update, hard_update

from model.net import QNetwork_1, QNetwork_2, ValueNetwork, GaussianPolicy, DeterministicPolicy
#from syscon_test_amcl_world import StageWorld

from A_syscon_gazebo_test_amcl_world import StageWorld
from model.sac import SAC

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Stage",
                    help='Environment name (default: Stage)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(\tau) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter \alpha determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust \alpha (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=200000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--laser_beam', type=int, default=512,
                    help='the number of Lidar scan [observation] (default: 512)')
parser.add_argument('--num_env', type=int, default=1,
                    help='the number of environment (default: 1)')
parser.add_argument('--laser_hist', type=int, default=3,
                    help='the number of laser history (default: 3)')
parser.add_argument('--act_size', type=int, default=2,
                    help='Action size (default: 2, translation, rotation velocity)')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch (default: 1)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')                    
args = parser.parse_args()

def run(comm, agent, policy_path, args):

    for i in range(100):
            
        # Get initial state
        scan1 = np.ones((512,), dtype=float) * 9
        scan2 = np.ones((512,), dtype=float) * 9
        scan3 = np.ones((512,), dtype=float) * 9

        for j in range(100):
            scan1[200 + j] = 10
            scan2[200 + j] = 10
            scan3[200 + j] = 10
            
        for j in range(30):
            scan1[220+j] = 2.0 - i*0.01
        for j in range(50):
            scan2[220+j] = 2.0 - i*0.01
        for j in range(50):
            scan3[220+j] = 2.0 - i*0.01


        frame1 = get_laser_observation(scan1)
        frame2 = get_laser_observation(scan2)
        frame3 = get_laser_observation(scan3)

        #print(frame)
        frame_stack = deque([frame1, frame2, frame3])
        
        goal = np.asarray([0, 50])
        speed = np.asarray([0.4, 0])
        state = [frame_stack, goal, speed]

        state_list = comm.gather(state, root=0)

        action = agent.select_action(state_list, evaluate=True)
        
        print(action)

def get_laser_observation(scan_s):
    scan = copy.deepcopy(scan_s)
    scan[np.isnan(scan)] = 10.0
    scan[np.isinf(scan)] = 10.0
    raw_beam_num = len(scan)
    sparse_beam_num = 512
    step = float(raw_beam_num) / sparse_beam_num
    sparse_scan_left = []
    index = 0.
    for x in xrange(int(sparse_beam_num / 2)):
        sparse_scan_left.append(scan[int(index)])
        index += step
    sparse_scan_right = []
    index = raw_beam_num - 1.
    for x in xrange(int(sparse_beam_num / 2)):
        sparse_scan_right.append(scan[int(index)])
        index -= step
    scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
    return scan_sparse / 10.0 - 0.5
    
if __name__ == '__main__':

    task_name = 'SIMUL_MCAL_B'
    log_path = './log/' + task_name

    # config log
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    output_file = log_path + '/epi_output.log'
    step_output_file = log_path + '/step_output.log'
    time_output_file = log_path + '/time_output.log'
    time_compute_output_file = log_path + '/time_compute_output.log'

    # config log
    logger = logging.getLogger('epi_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger.info('start_logging')

    logger_step = logging.getLogger('step_logger')
    logger_step.setLevel(logging.INFO)
    file_handler = logging.FileHandler(step_output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler.setLevel(logging.INFO)
    logger_step.addHandler(file_handler)
    logger_step.addHandler(stdout_handler)

    logger_step.info('start_logging')

    logger_time = logging.getLogger('time_logger')
    logger_time.setLevel(logging.INFO)
    file_handler = logging.FileHandler(time_output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler.setLevel(logging.INFO)
    logger_time.addHandler(file_handler)
    logger_time.addHandler(stdout_handler)

    logger_time.info('start_logging')

    logger_compute_time = logging.getLogger('time_logger')
    logger_compute_time.setLevel(logging.INFO)
    file_handler = logging.FileHandler(time_compute_output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler.setLevel(logging.INFO)
    logger_compute_time.addHandler(file_handler)
    logger_compute_time.addHandler(stdout_handler)

    comm = MPI.COMM_WORLD # There is one special communicator that exists when an MPI program starts, that contains all the processes in the MPI program. This communicator is called MPI.COMM_WORLD
    size = comm.Get_size() # The first of these is called Get_size(), and this returns the total number of processes contained in the communicator (the size of the communicator).
    rank = comm.Get_rank() # The second of these is called Get_rank(), and this returns the rank of the calling process within the communicator. Note that Get_rank() will return a different value for every process in the MPI program.
    print("MPI size=%d, rank=%d" % (size, rank))

    reward = None
    if rank == 0:
        policy_path = 'policy_test'
        #board_path = 'runs/r2_epi_0'
        # Agent num_frame_obs, num_goal_obs, num_vel_obs, action_space, args
        action_bound = [[0, 1], [-1, 1]] #### Action maximum, minimum values
        action_bound = spaces.Box(-1, +1, (2,), dtype=np.float32)
        agent = SAC(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        #'/w5a10_policy_epi_2000.pth'

        # static obstacle best policy
        file_policy = policy_path + '/syscon_6world_policy_epi_12900.pth' 
        #file_policy = policy_path + '/1028_policy_epi_2200.pth' 
        #file_policy = policy_path + '/stage_1027_policy_epi_1500.pth' 
        #file_policy = policy_path + '/20201022_policy_epi_4500.pth' 
        #file_policy = policy_path + '/1023_lsq_policy_epi_200.pth' 

        file_critic_1 = policy_path + '/syscon_6world_critic_1_epi_12900.pth'
        file_critic_2 = policy_path + '/syscon_6world_critic_2_epi_12900.pth'



        if os.path.exists(file_policy):
            print('###########################################')
            print('############Loading Policy Model###########')
            print('###########################################')
            state_dict = torch.load(file_policy)
            agent.policy.load_state_dict(state_dict)
        else:
            print('###########################################')
            print('############Start policy Training###########')
            print('###########################################')

        if os.path.exists(file_critic_1):
            print('###########################################')
            print('############Loading critic_1 Model###########')
            print('###########################################')
            state_dict = torch.load(file_critic_1)
            agent.critic_1.load_state_dict(state_dict)
            hard_update(agent.critic_1_target, agent.critic_1)

        else:
            print('###########################################')
            print('############Start critic_1 Training###########')
            print('###########################################')
    
        if os.path.exists(file_critic_2):
            print('###########################################')
            print('############Loading critic_2 Model###########')
            print('###########################################')
            state_dict = torch.load(file_critic_2)
            agent.critic_2.load_state_dict(state_dict)
            hard_update(agent.critic_2_target, agent.critic_2)
        else:
            print('###########################################')
            print('############Start critic_2 Training###########')
            print('###########################################')    

    else:
        agent = None
        policy_path = None
        
    try:
        run(comm=comm, agent=agent, policy_path=policy_path, args=args)
    except KeyboardInterrupt:
        pass
