from render_notebook import render_cart_pole

import time
import gym

class BasicCartpole():
    def __init__(self):

        self.env = gym.make("CartPole-v1", render_mode='human')
        print('obs :', self.obs)
        self.obs = self.env.reset()
        print('env.action_space :', self.env.action_space)
        return


    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,
    
    def render_scene(self):
        render_cart_pole(self.env, self.obs)
        return

    def update_obs(self, obs):
        self.obs = obs
        return
    
    @staticmethod
    def obs(self):
        return(self.obs)
    
    @staticmethod
    def env(self):
        return(self.env)

if __name__=="__main__":
    run = BasicCartpole()
    run.render_scene()
    time.sleep(5)
