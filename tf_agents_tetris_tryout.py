from __future__ import absolute_import, division, print_function

import numpy as np
from pathlib import Path
import tensorflow as tf
import logging
from datetime import datetime
import argparse


from engine import TetrisEngine


from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common



tf.compat.v1.enable_v2_behavior()
checkpoint_dir = Path("./checkpoints")

current_time = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
log_filename = 'logs/' + current_time
logging.basicConfig(filename=log_filename, level=logging.DEBUG)

train_log_dir = 'tensorboard/tf_agents_tryout/' + current_time + '/train'
test_log_dir = 'tensorboard/tf_agents_tryout/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# tf.config.gpu.set_per_process_memory_fraction(0.666)


num_iterations = 100000000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 10000  # @param {type:"integer"}

batch_size = 126  # @param {type:"integer"}
learning_rate = 0.0001  # @param {type:"number"}
log_interval = 500  # @param {type:"integer"}

num_eval_episodes = 50  # @param {type:"integer"}
eval_interval = 2000  # @param {type:"integer"}

width, height = 10, 20 # standard tetris friends rules

def perform_training(args):

  train_py_env = TetrisEngine(width, height)
  eval_py_env = TetrisEngine(width, height)
  train_env = tf_py_environment.TFPyEnvironment(train_py_env)
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


  fc_layer_params = (100, 50)
  action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
  num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

  # Define a helper function to create Dense layers configured with the right
  # activation and kernel initializer.
  def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

  # QNetwork consists of a sequence of Dense layers followed by a dense layer
  # with `num_actions` units to generate one q_value per available action as
  # it's output.
  dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
  q_values_layer = tf.keras.layers.Dense(
      num_actions,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.03, maxval=0.03),
      bias_initializer=tf.keras.initializers.Constant(-0.2))
  q_net = sequential.Sequential(dense_layers + [q_values_layer])

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  train_step_counter = tf.Variable(0)

  current_epsilon = 1.0
  epsilon_decay = 0.9999
  epsilon_min = 0.1

  def get_eplison():
    global current_epsilon
    pre_epsilon = current_epsilon
    current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
    return pre_epsilon

  agent = dqn_agent.DqnAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      q_network=q_net,
      gamma=0.95,
      optimizer=optimizer,
      epsilon_greedy=get_eplison,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=train_step_counter)

  agent.initialize()


  eval_policy = agent.policy
  collect_policy = agent.collect_policy

  random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                  train_env.action_spec())


  def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

      time_step = environment.reset()
      episode_return = 0.0

      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
      total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_max_length)

  def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

  def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
      collect_step(env, policy, buffer)

  collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

  # This loop is so common in RL, that we provide standard implementations. 
  # For more details see tutorial 4 or the drivers module.
  # https://github.com/tensorflow/agents/blob/master/docs/tutorials/4_drivers_tutorial.ipynb 
  # https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers



  # See also the metrics module for standard implementations of different metrics.
  # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics


  # Dataset generates trajectories with shape [Bx2x...]
  #TODO is This num_steps = 2 might not be optimal
  dataset = replay_buffer.as_dataset(
      num_parallel_calls=3, 
      sample_batch_size=batch_size, 
      num_steps=2).prefetch(3)

  iterator = iter(dataset)

  # (Optional) Optimize by wrapping some of the code in a graph using TF function.
  agent.train = common.function(agent.train)

  # Reset the train step
  agent.train_step_counter.assign(0)

  # Evaluate the agent's policy once before training.
  avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
  returns = [avg_return]
  best_return = -95



  checkpoint_dir = Path("./checkpoints")                       
  train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=20,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
  )


  for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
      logging.info('step = {0}: loss = {1}'.format(step, train_loss))
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss, step=step)

    if step % eval_interval == 0:
      avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
      print('step = {0}: Average Return = {1}'.format(step, avg_return))
      logging.info('step = {0}: Average Return = {1}'.format(step, avg_return))
      with test_summary_writer.as_default():
        tf.summary.scalar('avg return', avg_return, step=step)

      returns.append(avg_return)
      if avg_return > best_return + 2 and args.save_checkpoints:
        logging.info('Found better model, saving this model')
        train_checkpointer.save(train_step_counter)
        best_return = avg_return

  print("finished training")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--save_checkpoints", help="Set, if checkpoints should be used",
                    action="store_true")
  args = parser.parse_args()
  perform_training(args)