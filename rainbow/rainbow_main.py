# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

# import atari_py
import numpy as np
import torch
from tqdm import trange
import tensorflow as tf

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from rainbow_agent import Agent
# from env import Env
from engine import TetrisEngine
from rainbow_memory import ReplayMemory
# from test import test

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
session = InteractiveSession(config=config)


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
# parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=2, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')

parser.add_argument("--trainings_name", type=str, default="default", help="The time to wait after each env render.")

# Setup
args = parser.parse_args()

# Constants
num_eval_episodes = 50
eval_interval = 2000

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

# Logging Setup
current_time = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
train_log_dir = 'tensorboard/rainbow/' + args.trainings_name + "_" + current_time + '/train'
test_log_dir = 'tensorboard/rainbow/' + args.trainings_name + "_" + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
running_reward = 0
episode_reward_history = []


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)

def compute_avg_return(model, environment, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):
      state = environment.do_reset()
      episode_return = 0.0
      done = False
      counter = 0
      MAX_EPISODE_LENGTH = 200
      while not done and counter < MAX_EPISODE_LENGTH:
          # Take best action
          action = model.act(state)

          state, reward, done, _ = environment._step(action)
          episode_return += reward
          counter += 1
      total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return


# Environment
width, height = 10, 20 # standard tetris friends rules
train_env = TetrisEngine(width, height)
test_env = TetrisEngine(width, height)
action_space = train_env.get_action_count()

# Agent
dqn = Agent(args, train_env)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

  mem = load_memory(args.memory, args.disable_bzip_memory)

else:
  mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state = test_env.do_reset()

  next_state, _, done, _ = train_env._step(np.random.randint(0, action_space))
  val_mem.append(state, -1, 0.0, done)
  state = next_state
  T += 1

# Training loop
dqn.train()
T, done = 0, False
state = train_env.do_reset()
episode_reward = 0
total_cleared_lines = 0
for T in trange(1, args.T_max + 1):
  if done:
    with train_summary_writer.as_default():
      tf.summary.scalar('return', running_reward, step=T)
      tf.summary.scalar("cleared_lines_count", total_cleared_lines, step=T)
    state = train_env.do_reset()
    episode_reward = 0
    total_cleared_lines = 0

  if T % args.replay_frequency == 0:
    dqn.reset_noise()  # Draw a new set of noisy weights

  action = dqn.act(state)  # Choose an action greedily (with noisy weights)
  next_state, reward, done, additional_info = train_env._step(action)  # Step
  if args.reward_clip > 0:
    reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
  mem.append(state, action, reward, done)  # Append transition to memory

  episode_reward += reward
  episode_reward_history.append(episode_reward)
  if len(episode_reward_history) > 100:
    del episode_reward_history[:1]
  running_reward = np.mean(episode_reward_history)
  total_cleared_lines += additional_info["cleared_lines"]

  # Train and test
  if T >= args.learn_start:
    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

    if T % args.replay_frequency == 0:
      dqn.learn(mem)  # Train with n-step distributional double-Q learning

    if T % eval_interval == 0:
      dqn.eval()  # Set DQN (online network) to evaluation mode
      avg_return = compute_avg_return(dqn, test_env, num_eval_episodes)
      with test_summary_writer.as_default():
        tf.summary.scalar('avg return', avg_return, step=T)
      dqn.train()  # Set DQN (online network) back to training mode

    # Update target network
    if T % args.target_update == 0:
      dqn.update_target_net()

    # Checkpoint the network
    if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
      dqn.save(results_dir, 'checkpoint.pth')

  state = next_state