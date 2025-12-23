import os
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp  # 멀티프로세싱 모듈 추가

from utils.utils             import live_plot, save_model, load_model
from collections             import namedtuple

# Transition 정의
Transition = namedtuple('Transition',
                        ('state', 'action', 'prob', 'target_value', 'advantage'))
    
class PPO(object):
    def __init__(self, 
                #  Network
                 actor_critic_net:nn.Module, 
                 n_inputs:int, 
                 n_outputs:int, 
                #  Environment
                 workers:int,
                 environment,
                #  Common Learning Parameter
                 epoch:int=3,
                 batch:int=512,
                 max_episode:int=2000,
                 learning_rate:float=0.0005,
                 discount_factor:float=0.99,
                #  PPO Learning Parameter
                 advantage_lambda:float=0.95,
                 clip_epsilon:float=0.1,
                #  Torch
                 device:torch.device=torch.device('cpu'),
                #  Save Model
                 save_dir:str="./",
                 save_period:int=5,
                 ):
        """
        메인 process와 worker process로 나누어 진행

        메인 process: 
            - worker 생성 및 관리
            - worker에게 시뮬레이션 실행 명령 하달
            - 메인 네트워크 학습
            - worker 네트워크
        worker process:
            - 시뮬레이션 실행
            - 학습 데이터 생성
            - 학습 데이터 메인 process에게 batch 형태로 전달
        
        작업 pipeline
            1. 메인 process가 타겟네트워크 생성
            2. 메인 process가 worker process 생성
            3. worker process가 각각의 시뮬레이션 생성
            4. worker process가 각각의 실행네트워크 생성
            5. 학습 데이터 확득 (a - e 반복)
                a. 메인 process가 worker에게 실행 명령 및 타겟네트워크 파라미터 전달
                b. 각 worker는 보유한 실행 네트워크를 최신화
                c. 각 worker는 batch size만큼 시뮬레이션 실행 및 데이터 획득
                d. 각 worker는 시뮬레이션 데이터를 기반으로 GAE 계산
                e. batch 생성 순서대로 메인 process에 학습 데이터 전달 및 대기
                f. 메인 process는 worker로부터 학습 데이터 수신, 모든 데이터를 합쳐 학습 진행 (학습 데이터 크기 = worker num * batch size)
        """

        # 메인 네트워크 생성
        self.network_class = actor_critic_net
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.workers = workers
        self.environment = environment
        self.epoch = epoch
        self.batch = batch
        self.max_episode = max_episode
        self.discount_factor = discount_factor
        self.advantage_lambda = advantage_lambda
        self.clip_epsilon = clip_epsilon
        self.device = device
        self.save_dir = save_dir
        self.save_period = save_period

        self.actor_critic_net = self.network_class(self.n_inputs, self.n_outputs).to(self.device)
        self.optimizer = torch.optim.AdamW(self.actor_critic_net.parameters(), lr=learning_rate, amsgrad=True)

    def learn(self):
        # 워커 생성 및 파이프 연결
        workers = []
        pipes = []

        for i in range(self.workers):
            parent_conn, child_conn = mp.Pipe()
            # 워커 프로세스 생성 (워커는 CPU 연산을 주로 하도록 'cpu' 디바이스를 할당 (메인만 GPU))
            w = mp.Process(target=worker_process, 
                        args=(
                            i, 
                            child_conn, 
                            self.environment, 
                            self.network_class(self.n_inputs, self.n_outputs), 
                            torch.device('cpu'), 
                            self.batch,
                            self.discount_factor,
                            self.advantage_lambda
                            ))
            w.start()
            workers.append(w)
            pipes.append(parent_conn)
            
        print(f"Initialized {self.workers} workers.")

        # 저장 디렉토리 생성
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 학습 기록 변수
        total_steps = []
        step_done_set = []
        global_step = 0

        print("Training Start...")
        try:
            for episode in range(1, self.max_episode + 1):
                # 1. 모든 워커에게 수집 명령 전달 (현재 정책 가중치 포함)
                current_weights = {k: v.cpu() for k, v in self.actor_critic_net.state_dict().items()}
                
                for pipe in pipes:
                    pipe.send(('COLLECT', current_weights))

                # 2. 모든 워커로부터 데이터 수신 및 병합
                worker_transions = []
                batch_rewards = 0
                
                for pipe in pipes:
                    worker_trans, worker_reward = pipe.recv()
                    worker_transions.append(worker_trans)
                    batch_rewards += worker_reward
                
                # 3. 데이터 배치 생성 (Transition Unpacking) CPU -> GPU
                state_batch = torch.cat([w.state for w in worker_transions]).to(self.device)
                action_batch = torch.cat([w.action for w in worker_transions]).to(self.device)
                prob_old_batch = torch.cat([w.prob for w in worker_transions]).to(self.device)
                target_value_batch = torch.cat([w.target_value for w in worker_transions]).to(self.device)
                advantage_batch = torch.cat([w.advantage for w in worker_transions]).to(self.device)

                # 4. PPO 학습 진행 (Optimize)
                for _ in range(self.epoch):
                    # Actor: 현재 정책의 확률 계산
                    distribution = self.actor_critic_net.Actor(state_batch)
                    prob_batch = distribution.log_prob(action_batch)

                    # Ratio 계산
                    imp_ratio = torch.exp(prob_batch - prob_old_batch).unsqueeze(-1)

                    # Clipping
                    clip_advantage = torch.min(imp_ratio * advantage_batch, 
                                            torch.clamp(imp_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch)

                    # Loss 계산
                    actor_loss = -clip_advantage.mean()

                    # Critic Loss (MSE)
                    value_curr = self.actor_critic_net.Critic(state_batch)
                    critic_loss = nn.MSELoss()(target_value_batch, value_curr)

                    loss = actor_loss + critic_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # 5. 로깅 및 저장
                avg_reward = batch_rewards / self.workers # 워커 평균 보상 (단순 참고용)
                step_done_set.append(avg_reward)
                global_step += self.batch * self.workers
                
                print(f"Ep: {episode} | Step: {global_step} | Avg Reward (Batch): {avg_reward:.2f}")

                # 시각화 및 모델 저장
                if episode % self.save_period == 0:
                    current_avg = np.mean(step_done_set)
                    print(f"\tAvg reward from {episode - self.save_period} to {episode} episode:\t{np.mean(step_done_set):.2f}")
                    save_model(self.actor_critic_net, self.save_dir, "actor_critic", episode)
                    
                    total_steps.append(current_avg)
                    live_plot(total_steps, self.save_period) # live_plot 함수는 메인 스레드에서 실행
                    step_done_set = []

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            # 워커 종료 처리
            for pipe in pipes:
                pipe.send(('CLOSE', None))
            for w in workers:
                w.join()

    def execute(self, file_name:str, max_step:int=1024):
        sim = self.environment()
        actor_critic_net = load_model(self.actor_critic_net, self.save_dir, file_name).eval()
        
        total_reward = 0.0

        state_curr = sim.reset()
        state_curr = torch.tensor(state_curr, dtype=torch.float32, device=self.device)
        for _ in range(max_step):
            with torch.no_grad():
                distribution = actor_critic_net.Actor(state_curr.unsqueeze(0))
                most_likely_action = distribution.mean
                action = actor_critic_net.normalize_action(most_likely_action)
            state_next, reward, _ = sim.step(action)
            total_reward += reward

            print(action, f"\t{reward:.2f}\t{total_reward:.2f}")

            state_curr = torch.tensor(state_next, dtype=torch.float32, device=self.device)
            sim.render()

        sim.close()

def worker_process(worker_id, main_pipe, simulation, model_cls, device_worker, batch_size, discount_factor, advantage_lambda):
    """
    각 코어에서 환경을 실행하고 데이터를 수집 (학습 X)
    한번 실행할 때 마다 BATCH_SIZE만큼 실행, 학습 데이터를 한번에 모아 main 프로세스에 전달

    worker_id: 프로세스 ID
    main_pipe: 메인 프로세스와 통신하는 파이프
    """
    torch.set_num_threads(1)

    # 워커 내에서 환경 및 모델 독립 생성
    sim = simulation(None)

    local_model = model_cls.to(device_worker)
    local_model.eval() # 워커는 추론(Inference)만 수행

    # 통신 루프
    while True:
        # 1. 메인 프로세스로부터 명령 대기 (가중치 수신)
        cmd, data = main_pipe.recv()
        
        if cmd == 'CLOSE': # 종료 신호
            sim.close()
            break
            
        elif cmd == 'COLLECT': # 데이터 수집 신호
            # 최신 가중치 업데이트
            local_model.load_state_dict(data)
            
            memory_buffer = []
            total_reward = 0
            
            state_curr = sim.reset()
            state_curr = torch.tensor(state_curr, dtype=torch.float32, device=device_worker)

            # BATCH_SIZE 만큼 데이터 수집
            for _ in range(batch_size):
                with torch.no_grad():
                    distribution = local_model.Actor(state_curr.unsqueeze(0))
                    action = distribution.sample()
                    # log_prob 저장
                    log_prob = distribution.log_prob(action)
                    
                # 환경 Step
                action_np = local_model.normalize_action(action) # CPU로 변환
                state_next, reward, done = sim.step(action_np)
                total_reward += reward

                state_next_tensor = torch.tensor(state_next, dtype=torch.float32, device=device_worker)
                
                # 임시 버퍼 저장 (GAE 계산 전 Raw Data)
                memory_buffer.append({
                    'state': state_curr.unsqueeze(0),
                    'action': action,
                    'prob': log_prob,
                    'next_state': state_next_tensor.unsqueeze(0),
                    'reward': torch.tensor([reward], device=device_worker, dtype=torch.float32).unsqueeze(0),
                    'done': done
                })
                
                state_curr = state_next_tensor
                
                if done:
                    break

            # GAE Calculation (워커 내부에서 Advantage 계산)
            # 계산 속도를 위해 텐서로 변환
            states = torch.cat([x['state'] for x in memory_buffer])
            next_states = torch.cat([x['next_state'] for x in memory_buffer])
            rewards = torch.cat([x['reward'] for x in memory_buffer])
            
            with torch.no_grad():
                value_curr = local_model.Critic(states)
                value_next = local_model.Critic(next_states)
                
                # TD Target 계산
                td_target = rewards + discount_factor * value_next
                delta = (td_target - value_curr).cpu() # CPU에서 계산 (매트릭스 연산 부하 분산)

                # Advantage 계산
                # 주의: 워커별로 독립적인 Trajectory이므로 여기서 계산하는 것이 수학적으로 옳음
                powers = (discount_factor * advantage_lambda) ** torch.arange(len(value_curr), dtype=torch.float32)
                matrix = torch.tril(powers.view(-1, 1) / powers.view(1, -1))
                advantage = torch.mul(delta.repeat(1, len(value_curr)), matrix)
                advantage = torch.sum(advantage, dim=0).unsqueeze(-1)
            
            # 전송을 위한 데이터 패키징 (Transition NamedTuple 사용)
            # GPU 텐서는 공유 메모리 문제 발생 가능하므로 CPU로 옮겨서 전송 권장
            # states = states
            actions = torch.cat([x['action'] for x in memory_buffer])
            probs = torch.cat([x['prob'] for x in memory_buffer])
            target_values = td_target
            advantages = advantage
            
            transitions = Transition(
                state = states.cpu(),
                action = actions.cpu(),
                prob = probs.cpu(),
                target_value = target_values.cpu(),
                advantage = advantages.cpu()
            )
            # 결과 전송 (수집된 데이터, 누적 보상)
            main_pipe.send((transitions, total_reward))
