# Taken and modified from: https://raw.githubusercontent.com/keras-team/keras-io/master/examples/rl/deep_q_network_breakout.py
"""
Title: Deep Q-Learning for Atari Breakout
Author: [Jacob Chapman](https://twitter.com/jacoblchapman) and [Mathias Lechner](https://twitter.com/MLech20)
Date created: 2020/05/23
Last modified: 2020/06/17
Description: Play Atari Breakout with a Deep Q-Network.
"""
"""
## Introduction

This script shows an implementation of Deep Q-Learning on the
`BreakoutNoFrameskip-v4` environment.

This example requires the following dependencies: `baselines`, `atari-py`, `rows`.
They can be installed via:

```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
git clone https://github.com/openai/atari-py
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar .
python -m atari_py.import_roms .
```

### Deep Q-Learning

As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to an action. An agent will choose an action
in a given state based on a "Q-value", which is a weighted reward based on the
expected highest long-term reward. A Q-Learning Agent learns to perform its
task such that the recommended action maximizes the potential future rewards.
This method is considered an "Off-Policy" method,
meaning its Q values are updated assuming that the best action was chosen, even
if the best action was not chosen.

### Atari Breakout

In this environment, a board moves along the bottom of the screen returning a ball that
will destroy blocks at the top of the screen.
The aim of the game is to remove all blocks and breakout of the
level. The agent must learn to control the board by moving left and right, returning the
ball and removing all the blocks without the ball passing the board.

### Note

The Deepmind paper trained for "a total of 50 million frames (that is, around 38 days of
game experience in total)". However this script will give good results at around 10
million frames which are processed in less than 24 hours on a modern machine.

### References

- [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)
- [Deep Q-Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)
"""
"""
## Setup
"""

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import argparse
import curses
import os
import shutil
from engine import TetrisEngine
from time import sleep

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
session = InteractiveSession(config=config)

num_eval_episodes = 50
eval_interval = 2000

def get_env(width, height):
    env = TetrisEngine(width, height)
    return env

def compute_avg_return(model, environment, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):
        state = environment.do_reset()
        episode_return = 0.0
        done = False
        counter = 0
        MAX_EPISODE_LENGTH = 200
        while not done and counter < MAX_EPISODE_LENGTH:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state, dtype=tf.uint16)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

            state, reward, done, _ = environment._step(action)
            episode_return += reward
            counter += 1
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return

"""
## Implement the Deep Q-Network

This network learns an approximation of the Q-table, which is a mapping between
the states and actions that an agent will take. For every state we'll have four
actions, that can be taken. The environment provides the state, and the action
is chosen by selecting the larger of the four Q-values predicted in the output layer.

"""
def create_q_model(env: TetrisEngine):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=env.get_observation_shape())

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(4, 8, strides=1, activation="relu", padding="same")(inputs)
    layer2 = layers.Conv2D(8, 4, strides=1, activation="relu")(layer1)
    layer3 = layers.Conv2D(16, 4, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(128, activation="relu")(layer4)
    action = layers.Dense(env.get_action_count(), activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

# For CLI Animation
def render_env(env, screen, current_epsilon, step_count, using_model, total_cleared_lines):
    # Render
    screen.clear()
    screen.addstr(str(env))
    screen.addstr("Current cleared lines: {}\n".format(total_cleared_lines))
    screen.addstr("Current Epsilon: {:.3f}\n".format(current_epsilon))
    screen.addstr("Current Steps: {}\n".format(step_count))
    screen.addstr("Used Model?: {}\n".format(using_model))
    screen.refresh()

"""
## Train
"""
def perform_training(args):

    gamma = args.gamma  # Discount factor for past rewards
    epsilon = args.epsilon_max  # Epsilon greedy parameter
    epsilon_min = args.epsilon_min  # Minimum epsilon greedy parameter
    epsilon_max = args.epsilon_max  # Maximum epsilon greedy parameter
    epsilon_interval = epsilon_max - epsilon_min  # Rate at which to reduce chance of random action being taken
    batch_size = args.batch_size  # Size of batch taken from replay buffer
    max_steps_per_episode = args.max_steps_per_episode
    width, height = 10, 20 # standard tetris friends rules

    current_time = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    train_log_dir = 'tensorboard/keras_dqn/' + args.trainings_name + "_" + current_time + '/train'
    test_log_dir = 'tensorboard/keras_dqn/' + args.trainings_name + "_" + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    screen = None
    if args.render_env:
        screen = curses.initscr()
        curses.noecho()
        curses.cbreak()

    # The first model makes the predictions for Q-values which are used to
    # make a action.
    train_env = get_env(width, height)
    test_env = get_env(width, height)
    model = None
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = None
    best_avg_return = float('-inf')
    if args.load_model_from:
        model = keras.models.load_model(args.load_model_from)
        model_target = keras.models.load_model(args.load_model_from)
        # Read the avg return from the loaded models name
        best_avg_return = float(args.load_model_from.split("_")[-1].split(".")[0])
    else:
        model = create_q_model(train_env)
        model_target = create_q_model(train_env)
    model_target = create_q_model(train_env)
    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    step_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = args.epsilon_random_frames
    # Number of frames for exploration
    epsilon_greedy_frames = args.epsilon_greedy_frames
    # Maximum replay length
    max_memory_length = args.max_memory_length
    # Train the model after 4 actions
    update_after_actions = args.update_after_actions
    # How often to update the target network
    update_target_network = args.update_target_network
    # Using huber loss for stability
    loss_function = keras.losses.Huber()
    last_saved_model_path = None

     
    while True:  # Run until solved
        state = train_env.do_reset()
        episode_reward = 0
        total_cleared_lines = 0 
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            step_count += 1
            using_model = False
            # Use epsilon-greedy for exploration
            if step_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(train_env.get_action_count())
            else:
                # Predict action Q-values
                # From environment state
                using_model = True
                state_tensor = tf.convert_to_tensor(state, dtype=tf.uint16)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

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

            # Update every fourth frame and once batch size is over 32
            if step_count % update_after_actions == 0 and len(done_history) > batch_size:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = tf.convert_to_tensor([state_history[i] for i in indices], dtype=tf.uint16)
                state_next_sample = tf.convert_to_tensor([state_next_history[i] for i in indices], dtype=tf.uint16)
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, train_env.get_action_count())

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)
                # log the current loss    
                # with train_summary_writer.as_default():
                #    tf.summary.scalar('loss', loss[0], step=step_count)
                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, step_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if step_count % eval_interval == 0 and step_count > epsilon_random_frames:
                avg_return = compute_avg_return(model, test_env, num_eval_episodes)
                if avg_return > best_avg_return and args.save_model:
                    model_dir_path = "checkpoints/" + str(current_time) + "_" + args.trainings_name + "_{:.2f}.cpt".format(avg_return)
                    model.save(model_dir_path)
                    if last_saved_model_path and os.path.exists(last_saved_model_path):
                        shutil.rmtree(last_saved_model_path)
                    last_saved_model_path = model_dir_path
                    print("Saved new model to " + model_dir_path + ".")
                best_avg_return = max(best_avg_return, avg_return)
                with test_summary_writer.as_default():
                    tf.summary.scalar('avg return', avg_return, step=step_count)

            if done:
                with train_summary_writer.as_default():
                    tf.summary.scalar('return', running_reward, step=step_count)
                    tf.summary.scalar("cleared_lines_count", total_cleared_lines, step=step_count)
                break

            if args.render_env:
                render_env(train_env, screen, epsilon, step_count, using_model, total_cleared_lines)

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if running_reward > 100:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model", help="Whether to save the model with the given training name as a checkpoint once its average return improves.",
                    action="store_true")
    parser.add_argument("--load_model_from", help="The path to load the model from.",
                    type=str)
    parser.add_argument("--render_env", action="store_true",
                    help="To enable rendering the env after each training step")
    parser.add_argument("--render_env_sleep_time", type=float, default=0.15,
                    help="The time to wait after each env render.")
    parser.add_argument("--trainings_name", type=str, default="default",
                    help="The time to wait after each env render.")
    parser.add_argument("--epsilon_random_frames", type=int, default=50000,
                    help="Number of frames to take random action and observe output")
    parser.add_argument("--epsilon_greedy_frames", type=int, default=1000000,
                    help="Number of frames for exploration")
    parser.add_argument("--max_memory_length", type=int, default=100000,
                    help="Maximum replay length")
    parser.add_argument("--update_after_actions", type=int, default=4,
                    help="Train the model after update_after_actions many actions")
    parser.add_argument("--update_target_network", type=int, default=10000,
                    help="How often to update the target network")
    parser.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor for past rewards")
    parser.add_argument("--epsilon_min", type=float, default=0.1,
                    help="Minimum for the epsilon greed factor")
    parser.add_argument("--epsilon_max", type=int, default=1.0,
                    help="Maximum for the epsilon greed factor")
    parser.add_argument("--batch_size", type=int, default=32,
                    help="Size of batch taken from replay buffer")
    parser.add_argument("--max_steps_per_episode", type=int, default=10000,
                    help="Maximum steps an episode can have before resetting the environment")

    args = parser.parse_args()
    perform_training(args)

"""
## Visualizations
Before any training:
![Imgur](https://i.imgur.com/rRxXF4H.gif)

In early stages of training:
![Imgur](https://i.imgur.com/X8ghdpL.gif)

In later stages of training:
![Imgur](https://i.imgur.com/Z1K6qBQ.gif)
"""
