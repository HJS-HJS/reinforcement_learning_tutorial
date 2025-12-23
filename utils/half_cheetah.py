import numpy as np
import gymnasium as gym

class HalfCheetah():
    def __init__(self, visualize:str = 'human'):

        self.env = gym.make("HalfCheetah-v5", render_mode=visualize)

        self.N_INPUTS = self.env.observation_space.shape[0]
        self.N_OUTPUT = self.env.action_space.shape[0]
        return
    
    def render(self):
        return self.env.render()

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        state_next, reward, done, _, _ = self.env.step(action)
        return state_next, reward, done
    
    def close(self):
        return self.env.close()

    def test(self):
        for _ in range(1000):
            self.env.render()  # 시각화
            action = self.env.action_space.sample()  # 무작위 행동 샘플링
            obs, reward, done, info, _ = self.env.step(action)  # 환경에서 한 스텝 진행
            
            if done:
                obs = self.env.reset()  # 에피소드 종료 시 환경 초기화
                print('done')

        self.env.close()

if __name__=="__main__":
    run = HalfCheetah()
    run.test()
