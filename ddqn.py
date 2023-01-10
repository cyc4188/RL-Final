from utils import *

class DDQN(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed, layer_type="ff"):
        super(DDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size

        self.head = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        weight_init([self.head, self.ff_1])
    
    def forward(self, input):
        x = torch.relu(self.head(input))
        x = torch.relu(self.ff_1(x))
        ret = self.ff_2(x)
        return ret
