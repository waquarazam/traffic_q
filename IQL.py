import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.env import Edge,TrafficSimulatorEdge
import numpy as np
import random
from random import randint
from torch.autograd import Variable


class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, seed, fc1_unit=64,
                 fc2_unit = 64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        #self.soft = nn.Softmax(dim=1)


    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        #print(x)
        #print(type(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        #return self.soft(x)



name_to_code = {}
def encode_edges(edge_names):
    for idx in range(len(edge_names)):
        name = edge_names[idx]
        code = np.zeros(len(edge_names),dtype = np.double)
        code[idx] = np.double(1)
        name_to_code[name] = code

def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    =======
        local model (PyTorch model): weights will be copied from
        target model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                       local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

BATCH_SIZE = 1         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
DEFAULT_PORT = 8001
simulator = TrafficSimulatorEdge(0)
edge_names = simulator.edge_names
edges = simulator.edges
n_edges = len(edges)
agents=[]
seed = simulator.seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps = 0.1
DISCOUNT_FACTOR = 0.9

#optimizer = torch.optim.Adam(qnetwork_local.parameters(),lr=LR)
encode_edges(edge_names)
#print(name_to_code)
for edge_name in edge_names:
    edge = edges[edge_name]
    qnetwork_local = QNetwork(n_edges, len(edge.neighbour), seed).to(device)
    qnetwork_target = QNetwork(n_edges, len(edge.neighbour), seed).to(device)
    qnetwork_local = qnetwork_local.float()
    qnetwork_target = qnetwork_target.float()
    #print(len(edge.neighbour))
    edge.qnetwork_local = qnetwork_local
    edge.qnetwork_target = qnetwork_target
    optimizer = torch.optim.Adam(qnetwork_local.parameters(),lr=LR)
    edge.optimizer = optimizer

criterion = torch.nn.MSELoss()
        #self.qnetwork_local.train()
        #self.qnetwork_target.eval()
for index in range(100):
    actions = []

    states = []
    step_info = [] #tuple(edge,vehicle,state,action, act_val)
    for edge_name in edge_names:
        if ':' in edge_name:
            continue
        edge = edges[edge_name]
        local = edge.qnetwork_local
        target = edge.qnetwork_target
        local = local.float()
        target = target.float()
        target.eval()
        local.train()

        #print("edge name",edge_name)
        edge_states = simulator.get_state(edge)
        states.append(edge_states)
        edge_actions = []
        for state in edge_states:
            with torch.no_grad():
                #print("---------------------------------------------",state)
                var1 = Variable(torch.from_numpy(name_to_code[state[0]]),requires_grad=True)
                action_values = local(var1.float())
            local.train()
            #print(action_values)


            #Epsilon -greedy action selction
            act=0
            if random.random() > eps:
                act = np.argmax(action_values.cpu().data.numpy())
                #edge_actions.append(np.argmax(action_values.cpu().data.numpy()))
            else:
                act = randint(0, len(edge.neighbour)-1)
                #edge_actions.append(np.arange(len(edge.neighbour)))


            act = edge.neighbour[act]
            #print(edge.name, act)
            step_info.append((edge_name,state[1],state[0],act,action_values.cpu().data.numpy()))
            edge_actions.append(act)

        actions.append(edge_actions)
    obs,done = simulator.step(step_info) # #tuple(edge,vehicle,state,action, action_values,terminal,reward)
    #print(obs)

    for ob in obs:


        edge_name = ob[0]
        edge = edges[edge_name]
    #     = edges[idx]
        local = edge.qnetwork_local
        target = edge.qnetwork_target

        predicted_target = ob[4]
        terminal_state = ob[5]
        reward = ob[6]
        predicted_target = max(predicted_target)
        next_state = ob[2]
        criterion = torch.nn.MSELoss()
        local.train()
        target.eval()
        with torch.no_grad():
            var2 = Variable(torch.from_numpy(name_to_code[next_state]),requires_grad=True)
            label_next = max(qnetwork_target(var2.float()).cpu().data.numpy())

        print(label_next,predicted_target)
        target = reward + DISCOUNT_FACTOR*label_next
        loss = criterion(torch.tensor([predicted_target]),torch.tensor([target])).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #reward = rewards[idx]
        #action = actions[idx]
        #state = states[i]


        #predicted_targets = local(name_to_code[state]).gather(1,action)





states = np.array([0]*n_edges)

actions=[]

simulator.terminate()
