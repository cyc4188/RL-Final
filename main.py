from utils import *
from agent import *

def eval_runs(eps, frame):
    """
    Makes an evaluation run with the current epsilon
    """
    env = gym.make(game_name)
    reward_batch = []
    for i in range(5):
        state, _ = env.reset()
        rewards = 0

        while True:
            action = agent.get_action(state, eps)
            state, reward, done, _, _ = env.step(action)
            rewards += reward
            
            if done or rewards < -1000:
                break

        reward_batch.append(rewards)
    return np.mean(reward_batch) 

def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01):
    """
    Deep Munchausen Q-Learning.
    """
    scores = []
    scores_window: list[float] = []
    output_history = []
    eval_history = []
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    i_episode = 1
    state, _ = env.reset()
    score = 0                  
    for frame in range(1, frames+1):

        action = agent.get_action(state, eps)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame*(1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)
        
        if frame % 1000 == 0:
            eval_history.append(eval_runs(eps, frame))

        if score < -1000:
            done = True
        
        if done:
            scores_window.append(score)       # save most recent score
            scores_window = scores_window[-10:]
            scores.append(score)              # save most recent score
            output_history.append(np.mean(scores_window))
            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)), end="")
            if i_episode % 10 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,frame, np.mean(scores_window)))
            i_episode +=1 
            state, _ = env.reset()
            score = 0              

    return (output_history, eval_history)

if __name__ == "__main__":
    
    # game_name = "CartPole-v1"
    game_name = "Acrobot-v1"
    """
    DQN, MDQN
    """
    METHOD = "MDQN"
    seed = 114514 
    BUFFER_SIZE = 100000
    BATCH_SIZE = 8
    GAMMA = 0.99
    TAU = 1e-2
    LR = 1e-3
    LEARN_EVERY = 1
    lo = -1
    entropy_tau = 0.03
    alpha = 0.9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)


    np.random.seed(seed)
    env: gym.Env = gym.make(game_name)

    # env.seed(seed)
    action_size     = env.action_space.n
    state_size = env.observation_space.shape

    print("Action space: ", action_size)
    print("State space: ", state_size)

    agent = Agent(METHOD=METHOD,
                  state_size=state_size,    
                  action_size=action_size,
                  layer_size=256,
                  BATCH_SIZE=BATCH_SIZE, 
                  BUFFER_SIZE=BUFFER_SIZE, 
                  LR=LR, 
                  TAU=TAU, 
                  GAMMA=GAMMA, 
                  LEARN_EVERY=LEARN_EVERY, 
                  device=device, 
                  seed=seed,
                  alpha=alpha,
                  entropy_tau=entropy_tau,
                  lo=lo)



    # False: linear annealing
    # True:  eps fixed to 0
    eps_fixed = False

    output_history, eval_history = run(frames = 50000, eps_fixed=eps_fixed, eps_frames=5000, min_eps=0.025)
    
    save_file(game_name, METHOD, output_history)
    save_file(game_name, METHOD + '_eval', eval_history)
