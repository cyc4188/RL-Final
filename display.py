# from matplotlib.cbook import matplotlib
import matplotlib
from utils import *
import matplotlib.pyplot as plt

games = ['CartPole-v1', 'Acrobot-v1']

methods = ['MDQN', 'DQN']

matplotlib.use('TkAgg')
# draw 
for game in games:
    for method in methods:
        data = get_file(game, method)
        plt.plot(data)
    plt.legend(methods)
    plt.title(game)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

for game in games:
    for method in methods:
        data = get_file(game, method + '_eval')
        plt.plot(data)
    plt.legend(methods)
    plt.title(game)
    plt.xlabel('Episode')
    plt.ylabel('Reward (eval)')
    plt.show()
