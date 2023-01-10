from ddqn import *
from Buffer import *

class Agent():

    def __init__(self,
                 METHOD,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 LEARN_EVERY,
                 device,
                 seed,
                 alpha,
                 entropy_tau,
                 lo):
        """
        Initialize an Agent object.
        """
        self.METHOD = METHOD
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.LEARN_EVERY = LEARN_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.alpha = alpha
        self.entropy_tau = entropy_tau
        self.lo = lo

        self.Q_updates = 0

        # self.action_step = 4
        self.last_action = None
    
        # Q-Network
        self.qnetwork_local = DDQN(state_size, action_size,layer_size, seed).to(device)
        self.qnetwork_target = DDQN(state_size, action_size,layer_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, 1)
        
        # Initialize time step (for updating every LEARN_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every LEARN_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.LEARN_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1

    def get_action(self, state, eps=0.):
        """ returns actions for given state as per current policy. """

        state = np.array(state, dtype=np.float32)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
            action = np.argmax(action_values.cpu().data.numpy())
            self.last_action = action
            return action
        else:
            action = random.choice(np.arange(self.action_size))
            self.last_action = action 
            return action

    def learn(self, experiences):
        """ update the value parameters using given batch of experience tuples. """
        if self.METHOD == 'MDQN':
            return self.MDQN_learn(experiences)
        else :
            return self.DQN_learn(experiences)

    def DQN_learn(self, experiences):
        """DQN learning algorithm"""
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # Get predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        loss.backward()
        #clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # update target network 
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()            

    def MDQN_learn(self, experiences):
        """MDQN learning algorithm"""
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # Get predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach()
        # calculate entropy term with logsum 
        logsum = torch.logsumexp(\
                                (Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1))/self.entropy_tau , 1).unsqueeze(-1)

        tau_log_pi_next = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) - self.entropy_tau*logsum
        # target policy
        pi_target = F.softmax(Q_targets_next/self.entropy_tau, dim=1)
        Q_target = (self.GAMMA * (pi_target * (Q_targets_next-tau_log_pi_next)*(1 - dones)).sum(1)).unsqueeze(-1)
        
        # calculate munchausen addon with logsum trick
        q_k_targets = self.qnetwork_target(states).detach()
        v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
        logsum = torch.logsumexp((q_k_targets - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)
        log_pi = q_k_targets - v_k_target - self.entropy_tau * logsum
        munchausen_addon = log_pi.gather(1, actions)
        
        # calc munchausen reward:
        munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0))
        
        # Compute Q targets for current states 
        Q_targets = munchausen_reward + Q_target
        
        # Get expected Q values from local model
        q_k = self.qnetwork_local(states)
        Q_expected = q_k.gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) #mse_loss
        # Minimize the loss
        loss.backward()
        #clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # update target network 
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()            

    def soft_update(self, local_model, target_model):
        """Soft update model """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

