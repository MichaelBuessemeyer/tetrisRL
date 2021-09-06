"""
This is a simple random agent created to compare the results with random behavior. This is a modified copy of keras_dqn.py
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf
import argparse
import os
from engine import TetrisEngine


num_eval_episodes = 50
eval_interval = 2000
limit = 1000000

def get_env(width, height):
    env = TetrisEngine(width, height)
    return env

def compute_avg_return(environment, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):
        _state = environment.do_reset()
        episode_return = 0.0
        done = False
        counter = 0
        MAX_EPISODE_LENGTH = 200
        while not done and counter < MAX_EPISODE_LENGTH:
            # Take a random action
            action = np.random.choice(environment.get_action_count())
            # Take best action

            _state, reward, done, _ = environment._step(action)
            episode_return += reward
            counter += 1
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return


"""
## Train
"""
def perform_training(args):

    width, height = 10, 20 # standard tetris friends rules

    current_time = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    train_log_dir = 'tensorboard/random_agent/' + args.trainings_name + "_" + current_time + '/train'
    test_log_dir = 'tensorboard/random_agent/' + args.trainings_name + "_" + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    # Maximum replay length
    max_memory_length = args.max_memory_length
    max_steps_per_episode = args.max_steps_per_episode

    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    step_count = 0


    # The first model makes the predictions for Q-values which are used to
    # make a action.
    train_env = get_env(width, height)
    test_env = get_env(width, height)

    while step_count < limit:  # Run for 1 million
        state = train_env.do_reset()
        episode_reward = 0
        total_cleared_lines = 0 
        for timestep in range(1, max_steps_per_episode):
            step_count += 1
            using_model = False
            # Use epsilon-greedy for exploration
            action = np.random.choice(train_env.get_action_count())
            # Apply the sampled action in our environment
            state_next, reward, done, additional_info = train_env._step(action)
            total_cleared_lines += additional_info["cleared_lines"]

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if step_count % eval_interval == 0:
                avg_return = compute_avg_return(test_env, num_eval_episodes)
                with test_summary_writer.as_default():
                    tf.summary.scalar('avg return', avg_return, step=step_count)

            if done:
                with train_summary_writer.as_default():
                    tf.summary.scalar('return', running_reward, step=step_count)
                    tf.summary.scalar("cleared_lines_count", total_cleared_lines, step=step_count)
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainings_name", type=str, default="default",
                    help="The time to wait after each env render.")
    parser.add_argument("--max_memory_length", type=int, default=100000,
                    help="Maximum replay length")
    parser.add_argument("--max_steps_per_episode", type=int, default=10000,
                    help="Maximum steps an episode can have before resetting the environment")

    args = parser.parse_args()
    perform_training(args)
