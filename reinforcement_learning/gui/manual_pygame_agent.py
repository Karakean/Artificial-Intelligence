from time import sleep

from rl_base import Agent, ActionControl
import pygame

from pygame.locals import (
    K_ESCAPE,
    K_q,
    K_w,
    K_s,
    K_d,
    K_a,
    KEYDOWN,
    QUIT,
)


class QuitException(Exception):
    pass


class SimpleManualControl(ActionControl):

    def get_action(self, agent, observation):
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if (event.key == K_ESCAPE) | (event.key == K_q):
                        raise QuitException()
                    elif event.key == K_w:
                        action = 0
                    elif event.key == K_s:
                        action = 1
                    elif event.key == K_a:
                        action = 3
                    elif event.key == K_d:
                        action = 2
                elif event.type == QUIT:
                    raise QuitException()
            sleep(0.05)
        return action

    def get_instruction_string(self):
        return ["w - move up", "a - move left", "d - move right", "s - move down"]


class ManualPygameAgent(Agent):

    def __init__(self):
        super().__init__(name="Manual Pygame Agent")
        self.action_control = SimpleManualControl()

    def choose_action(self, observation):
        return self.action_control.get_action(self, observation)

    def get_instruction_string(self):
        return self.action_control.get_instruction_string()

    def save(self, save_path):
        pass

    def learn(self, observation, action, reward, new_observation, done):
        pass
