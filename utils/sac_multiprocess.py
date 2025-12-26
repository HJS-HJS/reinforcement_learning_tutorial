import os
import time
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.utils             import live_plot, save_model, load_model
from collections             import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class SAC(object):
    def __init__(self, 
                #  Network
                 actor_net_cls, 
                 q_net_cls,
                 n_inputs:int, 
                 n_outputs:int, 
                #  Environment
                 workers:int,
                 environment,
                #  Common Learning Parameter
                 batch_size:int=256,
                 max_steps:int=1024,
                 max_episode:int=2000,
                 memory_capacity:int=100000,
                 learning_rate:float=0.0005,
                 epoch:int=1,
                 discount_factor:float=0.99,
                #  PPO Learning Parameter
                 tau:float=0.005,
                 alpha_lr:float=0.001,
                 alpha:float=0.2,
                 target_entropy:float=-6.0,
                #  Torch
                 device:torch.device=torch.device('cpu'),
                #  Save Model
                 save_dir:str="./",
                 save_period:int=5,
                 ):

        # 메인 네트워크 생성
        self.actor_cls = actor_net_cls
        self.q_net_cls = q_net_cls
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.workers = workers
        self.environment = environment

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_episode = max_episode
        self.memory_capacity = memory_capacity
        self.lr = learning_rate
        self.epoch = epoch
        self.alpha_lr = alpha_lr
        self.gamma = discount_factor
        self.tau = tau
        self.alpha_init = alpha
        self.target_entropy = target_entropy
        
        self.device = device
        self.save_dir = save_dir
        self.save_period = save_period

        self.actor:nn.Module = self.actor_cls(n_inputs, n_outputs).to(self.device)
        self.q1:nn.Module = self.q_net_cls(n_inputs, n_outputs).to(self.device)
        self.q2:nn.Module = self.q_net_cls(n_inputs, n_outputs).to(self.device)
        self.target_q1:nn.Module = self.q_net_cls(n_inputs, n_outputs).to(self.device)
        self.target_q2:nn.Module = self.q_net_cls(n_inputs, n_outputs).to(self.device)
        
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Entropy
        self.log_alpha = torch.tensor(np.log(self.alpha_init), requires_grad=True, device=self.device)
        
        # Optimizers
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=self.lr)
        self.q1_optim = torch.optim.AdamW(self.q1.parameters(), lr=self.lr)
        self.q2_optim = torch.optim.AdamW(self.q2.parameters(), lr=self.lr)
        self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=self.alpha_lr)

        # Replay Buffer (Main Process Only)
        self.replay_buffer = [] 

        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        self.done_buffer = None

        self.buffer_ptr = 0
        self.buffer_size = 0

    def train(self):
        # --- Multi-processing Setup ---
        # Queue: Workers -> Main
        data_queue = mp.Queue(maxsize=1000) 
        weight_queues = [mp.Queue(maxsize=1) for _ in range(self.workers)]
        
        workers = []

        for i in range(self.workers):
            w = mp.Process(target=worker_process,
                           args=(i, 
                                 data_queue, 
                                 weight_queues[i], 
                                 self.environment, 
                                 self.actor_cls(self.n_inputs, self.n_outputs),
                                 self.max_steps,
                                 torch.device('cpu') # 워커는 CPU 사용
                                 ))
            w.start()
            workers.append(w)

        # --- Main Training Loop ---
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        total_steps = 0
        episode_rewards = []
        average_rewards = []
        processed_episodes = 0
        
        print("Training Start (Asynchronous SAC)...")
        try:
            while processed_episodes < self.max_episode:
                # 1. 데이터 수집 (Queue 비우기)
                while not data_queue.empty():
                    try:
                        transitions, ep_reward = data_queue.get_nowait()
                        self.push_to_buffer(transitions)
                        
                        if ep_reward is not None:
                            processed_episodes += 1
                            episode_rewards.append(ep_reward)
                            
                            # 저장 및 로깅
                            if processed_episodes % self.workers == 0:
                                step = processed_episodes // self.workers
                                avg_rew = np.mean(episode_rewards[-self.workers:])
                                print(f"{step}# Avg Reward: {avg_rew:.2f}")
                                if (processed_episodes // self.workers) % self.save_period == 0:
                                    average_rewards.append(np.mean(episode_rewards))
                                    episode_rewards = []

                                    print(f"{step}# Save Model (Avg Reward: {average_rewards[-1]:.2f})")
                                    live_plot(average_rewards, self.save_period)
                                    save_model(self.actor, self.save_dir, "actor_critic", step)

                    except queue.Empty:
                        break
                
                # 2. 학습 (데이터가 일정 이상 모였을 때)
                if self.buffer_size > self.batch_size * 2: # 초기 데이터 확보
                    for _ in range(self.epoch):
                        self.update_parameters()

                    # 3. 가중치 동기화 (일정 주기마다 워커에게 최신 가중치 전송)
                    current_actor_weights = {k: v.cpu() for k, v in self.actor.state_dict().items()}

                    for _w_q in weight_queues:
                        try:
                            _w_q.put_nowait(current_actor_weights)
                        except queue.Full:
                            pass
                else:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            print("Closing workers...")
            for _w_q in weight_queues:
                try:
                    while not _w_q.empty():
                        _w_q.get_nowait()
                except:
                    pass
                _w_q.put("CLOSE")

            for w in workers:
                w.join()
            data_queue.close()

    def execute(self, file_name:str, max_step:int=1024):
        sim = self.environment()
        actor_net = load_model(self.actor, self.save_dir, file_name).eval()
        
        total_reward = 0.0

        state_curr = sim.reset()
        state_curr = torch.tensor(state_curr, dtype=torch.float32, device=self.device).unsqueeze(0)
        for _ in range(max_step):
            with torch.no_grad():
                action, _ = actor_net(state_curr) 
                action_np = actor_net.normalize_action(action)

            state_next, reward, _ = sim.step(action_np)
            total_reward += reward

            print(action, f"\t{reward:.2f}\t{total_reward:.2f}")

            state_curr = torch.tensor(state_next, dtype=torch.float32, device=self.device).unsqueeze(0)
            sim.render()

        sim.close()

    def update_parameters(self):
        s, a, r, ns, d = self.sample_batch()
        
        # 1. Critic Update
        with torch.no_grad():
            next_action, next_log_prob = self.actor(ns)
            next_entropy = -self.log_alpha.exp() * next_log_prob.mean(dim=1).unsqueeze(1)
            
            next_target_q1 = self.target_q1(ns, next_action)
            next_target_q2 = self.target_q2(ns, next_action)

            target_min_q = torch.min(next_target_q1, next_target_q2)
            target_q = r + self.gamma * (1 - d) * (target_min_q + next_entropy)

        # learn Q
        current_q1 = self.q1(s, a)
        current_q2 = self.q2(s, a)
        
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()
        
        # 2. Actor Update
        action_new, log_prob_new = self.actor(s)
        entropy = (-self.log_alpha.exp() * log_prob_new.mean(dim=1)).unsqueeze(1)
        
        q1_new = self.q1(s, action_new)
        q2_new = self.q2(s, action_new)
        min_q_new = torch.min(q1_new, q2_new)
        
        actor_loss = -entropy - min_q_new
        
        self.actor_optim.zero_grad()
        actor_loss.mean().backward()
        self.actor_optim.step()
        
        # 3. Alpha Update
        alpha_loss = -(self.log_alpha.exp() * (log_prob_new.detach() + self.target_entropy)).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.mean().backward()
        self.alpha_optim.step()
        
        # 4. Soft Update Targets
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def sample_batch(self):
        """버퍼에서 배치 샘플링"""
        indices = np.random.choice(self.buffer_size, self.batch_size, replace=False)

        # (s, a, r, s', d)
        state = self.state_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        next_state = self.next_state_buffer[indices]
        done = self.done_buffer[indices]

        return state, action, reward, next_state, done
    
    def push_to_buffer(self, transitions):
        """워커로부터 받은 데이터를 버퍼에 저장"""

        state, action, reward, next_state, done = transitions

        if self.state_buffer is None:
            self.state_buffer = self._gen_empty_buffer(state.size())
            self.action_buffer = self._gen_empty_buffer(action.size())
            self.reward_buffer = self._gen_empty_buffer(reward.size())
            self.next_state_buffer = self._gen_empty_buffer(next_state.size())
            self.done_buffer = self._gen_empty_buffer(done.size())

        data_size = len(state)
        buffer_end_ptr = self.buffer_ptr + data_size

        if buffer_end_ptr < self.memory_capacity:
            self.state_buffer[self.buffer_ptr:buffer_end_ptr] = state.to(self.device)
            self.action_buffer[self.buffer_ptr:buffer_end_ptr] = action.to(self.device)
            self.reward_buffer[self.buffer_ptr:buffer_end_ptr] = reward.to(self.device)
            self.next_state_buffer[self.buffer_ptr:buffer_end_ptr] = next_state.to(self.device)
            self.done_buffer[self.buffer_ptr:buffer_end_ptr] = done.to(self.device)
        else:
            buffer_end_ptr %= self.memory_capacity
            _idx = self.memory_capacity - self.buffer_ptr

            self.state_buffer[self.buffer_ptr:] = state[:_idx].to(self.device)
            self.action_buffer[self.buffer_ptr:] = action[:_idx].to(self.device)
            self.reward_buffer[self.buffer_ptr:] = reward[:_idx].to(self.device)
            self.next_state_buffer[self.buffer_ptr:] = next_state[:_idx].to(self.device)
            self.done_buffer[self.buffer_ptr:] = done[:_idx].to(self.device)

            self.state_buffer[:buffer_end_ptr] = state[_idx:].to(self.device)
            self.action_buffer[:buffer_end_ptr] = action[_idx:].to(self.device)
            self.reward_buffer[:buffer_end_ptr] = reward[_idx:].to(self.device)
            self.next_state_buffer[:buffer_end_ptr] = next_state[_idx:].to(self.device)
            self.done_buffer[:buffer_end_ptr] = done[_idx:].to(self.device)

        self.buffer_ptr = buffer_end_ptr
        if self.buffer_size < self.buffer_ptr:
            self.buffer_size = self.buffer_ptr
        else:
            self.buffer_size = self.memory_capacity
    
    def _gen_empty_buffer(self, size):
        _size = list(size)
        _size[0] = self.memory_capacity
        return torch.zeros(_size, device=self.device)

def worker_process(worker_id:int, data_queue:mp.Queue, weight_queue:mp.Queue, env_cls, actor_model:nn.Module, max_step:int, device):
    """
    SAC Worker Process
    - 지속적으로 환경을 실행하며 (s, a, r, s', d)를 Queue에 넣음
    - Pipe를 polling하여 새로운 가중치가 오면 업데이트
    """
    torch.set_num_threads(1)
    
    sim = env_cls(None)
    # sim = env_cls()
    actor_model = actor_model.to(device)
    actor_model.eval()
    
    local_buffer = []
    
    while True:
        # 1. 가중치 업데이트 확인 (Non-blocking)
        if not weight_queue.empty():
            data = weight_queue.get_nowait()
            if isinstance(data, str) and data == 'CLOSE':
                break
            else:
                # Dictionary 형태의 가중치 수신
                actor_model.load_state_dict(data)
            
        # 2. 환경 실행
        state = sim.reset()

        episode_reward = 0
        step_done = False

        state_buffer = []
        action_buffer = []
        reward_buffer = []
        next_state_buffer = []
        done_buffer = []
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for step in range(1, max_step + 1):
            # Action Sampling
            with torch.no_grad():
                # Actor에서 action, logprob 추출
                action, _ = actor_model(state_tensor) 
                action_np = actor_model.normalize_action(action)
            
            next_state, reward, done = sim.step(action_np)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward += reward

            # (s, a, r, s', d) 저장
            state_buffer.append(state_tensor)
            action_buffer.append(action)
            reward_buffer.append(torch.tensor([reward], dtype=torch.float32, device=device).unsqueeze(0))
            next_state_buffer.append(next_state_tensor)
            done_buffer.append(torch.tensor([1 if done else 0], dtype=torch.float32, device=device).unsqueeze(0))

            state_tensor =  next_state_tensor

            step_done = done
            if step == max_step:
                step_done = True

            # SAC의 빠른 학습을 위해 매 스텝 보내는 방식 대신, 작은 청크(chunk) 단위 전송
            if len(state_buffer) >= 32 or step_done:
                local_buffer = [
                    torch.cat(state_buffer),
                    torch.cat(action_buffer),
                    torch.cat(reward_buffer),
                    torch.cat(next_state_buffer),
                    torch.cat(done_buffer),
                ]
                data_to_send = (local_buffer, episode_reward if step_done else None)
                data_queue.put(data_to_send)

                state_buffer = []
                action_buffer = []
                reward_buffer = []
                next_state_buffer = []
                done_buffer = []
            
            if done:
                break
                
    sim.close()
