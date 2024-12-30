import numpy as np
import gym

class HalfCheetah():
    def __init__(self, visualize:str = 'human'):

        self.env = gym.make("HalfCheetah-v4", render_mode=visualize)
        self.obs = self.env.reset()
        return

    def render(self):
        for _ in range(1000):
            self.env.render()  # 시각화
            action = self.env.action_space.sample()  # 무작위 행동 샘플링
            obs, reward, done, info, _ = self.env.step(action)  # 환경에서 한 스텝 진행
            
            if done:
                obs = self.env.reset()  # 에피소드 종료 시 환경 초기화
                print('done')

        self.env.close()
    
    def step(self, action):
        state_next, reward, done, _, _ = self.env.step(action)
        return state_next, reward, done

if __name__=="__main__":
    run = HalfCheetah()
    run.render()
