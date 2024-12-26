import time
import numpy as np
from PIL import Image, ImageDraw
from pynput.keyboard import Key, Listener

import gym

class CarRacing():
    def __init__(self, visualize:str = 'human'):

        self.env = gym.make("CarRacing-v2", render_mode=visualize)
        self.obs = self.env.reset()
        return

    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,

    def keyboard_input(self):
        key_pressed = {"left": 0, "right": 0, "acc": 0, "brake": 0}

        def on_press(key):
            if key == Key.left:
                key_pressed["left"] = 1
            elif key == Key.right:
                key_pressed["right"] = 1
            elif key == Key.up:
                key_pressed["acc"] = 1
            elif key == Key.down:
                key_pressed["brake"] = 1

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
        env = gym.make("CarRacing-v2", render_mode="human")
        obs, info = env.reset()

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()

        done = False
        while not done:
            # 키 상태에 따라 액션 설정
            action = [
                key_pressed["right"] - key_pressed["left"],  # 스티어링
                key_pressed["acc"],  # 가속
                key_pressed["brake"],  # 브레이크
            ]
            obs, reward, done, truncated, info = env.step(action)
        env.close()
        listener.stop()

    def update_obs(self, obs):
        self.obs = obs
        return
    
    @staticmethod
    def obs(self):
        return(self.obs)
    
    @staticmethod
    def env(self):
        return(self.env)
    
try:
    from pyglet.gl import gl_info
    openai_cart_pole_rendering = True   # 문제없음, OpenAI 짐의 렌더링 함수를 사용합니다
except Exception:
    openai_cart_pole_rendering = False  # 가능한 X 서버가 없다면, 자체 렌더링 함수를 사용합니다

if __name__=="__main__":
    run = CarRacing()
    run.env.render()
    run.keyboard_input()
    time.sleep(5)
