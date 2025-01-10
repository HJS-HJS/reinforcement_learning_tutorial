import cv2
import copy
import time
import numpy as np
from pynput.keyboard import Key, Listener
import matplotlib.pyplot as plt

import gym

class CarRacing():
    def __init__(self, visualize:str = 'human', frame_skip:int = 4):

        self.env = gym.make("CarRacing-v2", render_mode=visualize)
        self.obs = self.env.reset()
        self.frame = frame_skip
        self._buffer_size = 5
        self.reward_buffer = [True] * self._buffer_size
        self.local_state = [0] * int(self.frame)
        return

    def step(self, action):
        step_reward = 0.0
        is_green = False
        for _ in range(self.frame):
            state_next, reward, done, _, _ = self.env.step(action)
            if self._green_area_penalty(state_next): is_green = True
            if (not is_green) and (reward > 1): step_reward += 2.
        
        self.local_state.pop(0)
        self.local_state.append(self._preprocessing(state_next))

        # Finish step
        self.reward_buffer.pop(0)
        self.reward_buffer.append(not is_green)
        if not any(self.reward_buffer):
            done = True

        return np.array(self.local_state), step_reward, done
    
    def reset(self):
        self.env.reset()
        self.reward_buffer = [True] * self._buffer_size
        for _ in range(50):
            _ = self.env.step([0.,0.,0.])

        for _ in range(self.frame):
            state_next, _, _, _, _ = self.env.step([0.,0.,0.])
            self.local_state.pop(0)
            self.local_state.append(self._preprocessing(state_next))

        return np.array(self.local_state)
    
    def _preprocessing(self, img):
        # Hide progress bar to avoid casuality problem
        crop_img = img[:-12, 6:-6]
        # RGB to gray
        gray_img = np.dot(crop_img[..., :], [0.299, 0.587, 0.114])
        # Normalize image to [-1,1]
        prepro_img = 2 * (gray_img / 255.0) - 1
        return prepro_img.T
    
    def _green_area_penalty(self, img):
        # car area  img[60:-12, 38:-38]
        # front     img[60:65, 42:-42]
        # left      img[65:77, 43:46]
        # right     img[65:77, -46:-43]
        def _is_penalty(area):
            if np.mean(area[:,:1]) > 125: return True #penalty
            else:                         return False#reward
        return (_is_penalty(img[65:77, 43:46]) and _is_penalty(img[65:77, -46:-43]))

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
    run = CarRacing()
    run.env.render()
    run.keyboard_input()
    time.sleep(5)
