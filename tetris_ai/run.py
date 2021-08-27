import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
from tetris_ai.dqn_agent import DQNAgent
from tetris_ai.tetris import Tetris
from datetime import datetime
from statistics import mean, median
from tqdm import tqdm
        
def getDQNAgent(env, epsilon_stop_episode = 1500):
    activations = ['relu', 'relu', 'linear']
    replay_start_size = 2000
    discount = 0.95
    n_neurons = [32, 32]
    mem_size = 20000
    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)
    return agent

# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 50000
    max_steps = 1000
    mem_size = 20000
    batch_size = 512
    epochs = 1
    render_every = False
    log_every = 50
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    best_mean = 0.0

    agent = getDQNAgent(env)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            with train_summary_writer.as_default():
                tf.summary.scalar('avg_score', avg_score, step=episode)
                tf.summary.scalar('min_score', min_score, step=episode)
                tf.summary.scalar('max_score', max_score, step=episode)

            if avg_score > best_mean:
                agent.persist("checkpoints/tetris-ai-best.cpt")
                best_mean = avg_score

if __name__ == "__main__":
    dqn()
