import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.env import Edge,TrafficSimulatorEdge
import numpy as np
import random
from random import randint
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

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
        #code = np.zeros(len(edge_names),dtype = np.double)
        #code[idx] = np.double(1)
        code = idx+1
        name_to_code[name] = code

def soft_update(local_model, target_model, tau):
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
LR = 5e-4               # learning rate 5e-4
UPDATE_EVERY = 15        # how often to update the network
DEFAULT_PORT = 8001
simulator = TrafficSimulatorEdge(110)
edge_names = simulator.edge_names
edges = simulator.edges
n_edges = len(edges)
agents=[]
seed = simulator.seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps = 1
DISCOUNT_FACTOR = 0.9

#optimizer = torch.optim.Adam(qnetwork_local.parameters(),lr=LR)
encode_edges(edge_names)
print(name_to_code)
#print(name_to_code)
for edge_name in edge_names:
    edge = edges[edge_name]
    #qnetwork_local = QNetwork(n_edges, len(edge.neighbour), seed).to(device)
    #qnetwork_target = QNetwork(n_edges, len(edge.neighbour), seed).to(device)
    if(edge_name == "nt22_np14"):
        print("net size", len(edge.neighbour)+1)
    qnetwork_local = QNetwork(len(edge.neighbour) + 1, len(edge.neighbour), seed).to(device)
    qnetwork_target = QNetwork(len(edge.neighbour) + 1, len(edge.neighbour), seed).to(device)
    #print(qnetwork_local.state_dict())
    #print("----------------------------------------------------------------------")
    #print(qnetwork_target.state_dict())
    #exit()
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
done_episode = False
done_epoch = False
for epoch in range(simulator.no_of_epoch):
    accumulated_reward_list=[]
    for episode in range(simulator.no_episode_per_epoch):
        accumulated_reward=0
        for episode_step in range(simulator.no_of_episode_step):
            actions = []


            states = []
            step_info = [] #tuple(edge,vehicle,state,action, act_val)
            for edge_name in edge_names:
                #print("edge_name IQL", edge_name)
                #print()
                if ':' in edge_name:
                    continue
                edge = edges[edge_name]
                #print("edge",edge)
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
                #print("edge_states", edge_states)
                for state in edge_states:
                    #with torch.no_grad():
                        #print("---------------------------------------------",state)
                    s = state[0]
                    s[0] = name_to_code[s[0]]

                    var1 = Variable(torch.FloatTensor(s),requires_grad=True)
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

                    index = torch.tensor(act)
                    #print(action_values[0])
                    act = edge.neighbour[act]
                    #print(edge.name, act)
                    #print("state IQL", state)

                    #print(action_values)
                    #print()
                    #exit(0)
                    step_info.append((edge_name,state[1],state[0],act,action_values[index]))
                    edge_actions.append(act)

                actions.append(edge_actions)
            #print("step_info IQL",step_info)
            #print("--------------------------------------------------------------------------------------------------------------------------")
            #for x in step_info:
            #    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            #    print(x)
            #    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            #print("--------------------------------------------------------------------------------------------------------------------------")
            obs,done = simulator.step(step_info) # #tuple(edge,vehicle,state,action, action_values,terminal,reward)
            #print(obs)

            for ob in obs:


                edge_name = ob[0]
                edge = edges[edge_name]
                #print("edge_name", edge_name)
                #print("n size", len(edge.neighbour))
            #     = edges[idx]
                local = edge.qnetwork_local
                target = edge.qnetwork_target

                predicted_target = ob[4]
                #print(predicted_target)
                terminal_state = ob[5]
                reward = ob[6]
                accumulated_reward = accumulated_reward+reward
                #predicted_target = torch.max(predicted_target)
                #print(predicted_target)
                next_state = ob[2]
                criterion = torch.nn.MSELoss()
                local.train()
                target.eval()
                target_val = 0
                with torch.no_grad():
                    #print(next_state)
                    s = next_state
                    #print(s)
                    var2 = Variable(torch.FloatTensor(s),requires_grad=True)
                    label_next = torch.max(target(var2.float()))
                    #print(predicted_target,label_next)
                    target_val = reward + DISCOUNT_FACTOR*label_next*(1-terminal_state)

                #print(label_next,predicted_target)

                #print("predicted_target: ", predicted_target)
                #print("target: ", target)
                #print("24324  ",predicted_target, target_val)
                loss = criterion(predicted_target,target_val).to(device)
                #print(loss)
                #print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #reward = rewards[idx]
                #action = actions[idx]
                #state = states[i]


                #predicted_targets = local(name_to_code[state]).gather(1,action)
                soft_update(edge.qnetwork_local,edge.qnetwork_target,TAU)

            if episode_step%UPDATE_EVERY==0:
                print("updating target ", episode_step)
                for edge_name in edge_names:
                    edge = edges[edge_name]
                    #edge.qnetwork_target.load_state_dict(edge.qnetwork_local.state_dict())
                    soft_update(edge.qnetwork_local,edge.qnetwork_target,TAU)
                    edge.qnetwork_target.load_state_dict(edge.qnetwork_local.state_dict())




            if done or episode_step==simulator.no_of_episode_step-1:
                print("episode: ", episode)
                print("no of vehicles: ", simulator.get_no_of_vehicles())
                for edge_name in edge_names:
                    edge = edges[edge_name]

                    soft_update(edge.qnetwork_local,edge.qnetwork_target,TAU)
                    edge.qnetwork_target.load_state_dict(edge.qnetwork_local.state_dict())
                eps = eps * 0.99
                    #print(edge.qnetwork_target.state_dict())
                print("accumulated_reward",accumulated_reward)
                print("eps", eps)
                simulator.terminate()
                print("sim terminated")
                simulator.reset()
                accumulated_reward_list.append(accumulated_reward)
                break
        print(range(1,len(accumulated_reward_list)+1))
        print(accumulated_reward_list)
        if(episode%5==0):
            plt.plot(range(1,len(accumulated_reward_list)+1),accumulated_reward_list)
            plt.ylabel('reward')
            plt.xlabel("episode")
            plt.savefig("update after 10 high exploration eps decay .9 LR 5e-4 approach epoch"+str(epoch)+"episode"+str(episode)+'.png')


states = np.array([0]*n_edges)

actions=[]
