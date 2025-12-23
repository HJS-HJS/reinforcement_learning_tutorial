import cv2
import copy
import time
import numpy as np
from pynput.keyboard import Key, Listener
import matplotlib.pyplot as plt

import gymnasium as gym

class CarRacing():
    def __init__(self, visualize:str = 'human', frame_skip:int = 4, state_size:int=3):

        self.env = gym.make("CarRacing-v3", render_mode=visualize)
        self.obs = self.env.reset()
        self.frame = frame_skip
        self.local_state = [0] * int(state_size)

        self.state_size = state_size

        self.N_INPUTS = state_size
        self.N_OUTPUT = self.env.action_space.shape[0]

        return

    def render(self):
        return self.env.render()
    
    def step(self, action):
        step_reward = 0.0
        is_done = False
        for _ in range(self.frame):
            state_next, reward, done, _, _ = self.env.step(action)
            step_reward += reward
            if done: is_done = True 
            
        self.local_state.pop(0)
        self.local_state.append(self._preprocessing(state_next))

        return np.array(self.local_state), np.clip(step_reward, -1.0, 1.0), is_done
    
    def reset(self):
        self.env.reset()
        for _ in range(50):
            _ = self.env.step(np.zeros(3))

        for _ in range(len(self.local_state)):
            state_next, _, _, _, _ = self.env.step(np.zeros(3))
            self.local_state.pop(0)
            self.local_state.append(self._preprocessing(state_next))

        return np.array(self.local_state)
    
    def close(self):
        return self.env.close()
    
    def _preprocessing(self, img):
        # Hide progress bar to avoid casuality problem
        crop_img = img[:-12, 6:-6]
        gray_img = np.dot(crop_img[..., :], [0.299, 0.587, 0.114]) # RGB to gray
        prepro_img = (gray_img / 255.0) * 2.0 - 1.0 # Normalize image to [-1,1]
        return prepro_img.T
    
    @staticmethod
    def env(self):
        return(self.env)
    
    def keyboard_input(self):
        key_pressed = {"left": 0, "right": 0, "acc": 0, "brake": 0}

        def on_press(key):
            if key == Key.left:
                key_pressed["left"] = 0.5
            elif key == Key.right:
                key_pressed["right"] = 0.5
            elif key == Key.up:
                key_pressed["acc"] = 0.2
            elif key == Key.down:
                key_pressed["brake"] = 0.2

        def on_release(key):
            if key == Key.left:
                key_pressed["left"] = 0
            elif key == Key.right:
                key_pressed["right"] = 0
            elif key == Key.up:
                key_pressed["acc"] = 0
            elif key == Key.down:
                key_pressed["brake"] = 0
            if key == Key.esc:
                return False  # 종료

        # 환경 초기화
        _ = self.reset()

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()

        done = False
        while True:
            while not done:
                # 키 상태에 따라 액션 설정
                action = [
                    key_pressed["right"] - key_pressed["left"],  # 스티어링
                    key_pressed["acc"],  # 가속
                    key_pressed["brake"],  # 브레이크
                ]
                obs, reward, done = self.step(action)
            self.reset()

if __name__=="__main__":
    run = CarRacing2()
    run.env.render()
    run.keyboard_input()
    time.sleep(5)
