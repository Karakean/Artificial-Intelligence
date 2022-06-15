import os
from envs.frozen_lake import FrozenLake
from gui.main_pygame import main_pygame
from gui.manual_pygame_agent import ManualPygameAgent, SimpleManualControl
from rl_alg.q_agent import QAgent
from rl_alg.epsilon_greedy_strategy import EpsilonGreedyStrategy

from glob import glob
import datetime

if __name__ == '__main__':

    render = False
    mode = 'test'  # 'train' or 'test'

    agent = 'q_learning'            # 'q_learning' or 'manual'
    control = 'eps-greedy'          # 'eps-greedy' or 'manual'


    # TODO znajdź dobre wartości hiperparametrów
    # hyperparams
    lr = 0.1          # współczynnik uczenia (learning rate)
    gamma = 0.99           # współczynnik dyskontowania
    eps_start = 1.0       # epsilon początkowe
    eps_decay = 0.001       # wartość, o którą zmniejsza się epsilon w każdym kroku
    eps_end = 0.2         # końcowa wartość epsilon, poniżej którego już nie jest zmniejszane

    num_episodes = 500

    examined_env = FrozenLake()

    print(f"Env name: {examined_env.__class__.__name__}")
    print(f"Mode: {mode}")

    if agent == 'manual':
        agent = ManualPygameAgent()

    elif agent == 'q_learning':
        if control == 'eps-greedy':
            control = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
        elif control == 'manual':
            control = SimpleManualControl()
        else:
            raise ValueError("Not supported control: should be either 'eps-greedy' or 'manual'")
        agent = QAgent(16, 4, action_control=control, lr=lr, gamma=gamma)

    else:
        raise ValueError("Not supported agent: should be either 'q_learning' or 'manual'")

    # saving setup
    date = 'run-{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()).replace(':', '-')
    save_path_dir = '/'.join(['saved_models', examined_env.name, agent.name, date])
    save_path = save_path_dir + '/model'

    def get_prev_run_model(base_dir):
        dirs = glob(os.path.dirname(base_dir) + '/*')
        dirs.sort(reverse=True)
        return dirs[0] + '/model.npy'

    if mode == 'test':
        state_path = get_prev_run_model(save_path_dir)
        print(f"Testing model from latest run.")
        print(f"\tLoading agent state from {state_path}")
        agent.load(state_path)
        agent.action_control = EpsilonGreedyStrategy(0, 0, 0)
        print(f"TEST MODE, greedy action selection, eps={agent.action_control.epsilon}")

    main_pygame(examined_env, agent, save_path=save_path, render=render,
                num_episodes=num_episodes, test_mode=(mode == 'test'))
