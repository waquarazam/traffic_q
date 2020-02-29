
import logging
import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci
import xml.etree.cElementTree as ET
from large_grid.data.build_file import gen_rou_file

DEFAULT_PORT = 8001
SEC_IN_MS = 1000

# hard code real-net reward norm
REALNET_REWARD_NORM = 20

class Edge:
    """docstring for Edge."""

    def __init__(self, name, frm, to):
        self.name = name
        self.neighbour = None
        self.qnetwork_local = None
        self.qnetwork_target = None
        self.optimizer = None
        self.frm = frm
        self.to = to










class Node:
    def __init__(self, name, neighbour=[], control=False):
        self.control = control # disabled
        # self.edges_in = []  # for reward
        self.lanes_in = []
        self.ilds_in = [] # for state
        self.fingerprint = [] # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0 # wave and wait should have the same dim
        self.num_fingerprint = 0
        self.wave_state = [] # local state
        self.wait_state = [] # local state
        # self.waits = []
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1

class TrafficSimulatorEdge:
    def __init__(self, port=0):
        self.seed = 12
        self.episode_length_sec = 10
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.cur_episode = 0
        self.cur_step = 0
        self.no_of_episode_step = 50
        self.data_path = "./large_grid/data/"
        self.output_path = "./out/"
        self.name = "LargeGridEnv"
        self.agent = "IQL"
        self.peak_flow1 = 1100
        self.peak_flow2 = 925
        self.init_density = 0
        self.sim_thread = port
        self._init_sim(self.seed)
        self._init_edges()
        #self.terminate()

    def _init_edges(self):
        edges = {}
        count = 0
        for edge_name in self.sim.edge.getIDList():
            if ':' in edge_name:
                continue


            if (edge_name[0]!=':'):
                count = count + 1
            #print(edge_name)
            fromto = edge_name.split('_')
            #print("from ",frmto[0],"to ",frmto[1])
            edges[edge_name] = Edge(edge_name,fromto[0],fromto[1])
            #print(edge_name, neighbours)
        for edge_name in self.sim.edge.getIDList():
            if ':' in edge_name:
                continue
            neighbour = []
            edge = edges[edge_name]
            for tmp_edge_name in self.sim.edge.getIDList():
                if ':' in tmp_edge_name:
                    continue
                tmp_edge = edges[tmp_edge_name]
                if(edge.to == tmp_edge.frm):
                    neighbour.append(tmp_edge_name)
            edge.neighbour = neighbour
        self.edges = edges
        self.edge_names = sorted(list(edges.keys()))
        print(len(edges))
        print("count",count)

        s = 'Env: init %d edge information:\n' % len(self.edge_names)
        for edge in self.edges.values():
            s += edge.name + ':\n'
            s += '\tneighbour: %r\n' % edge.neighbour
            # s += '\tlanes_in: %r\n' % node.lanes_in
        logging.info(s)
        #self._init_action_space()
        #self._init_state_space()

    def get_state(self, edge):
        edge_name = edge.name
        state = []
        #print(edge_name)
        vehs = self.sim.edge.getLastStepVehicleIDs(edge_name)

        for veh in vehs:
            state.append((self.sim.vehicle.getRoute(veh)[-1],veh))
        return state


    def _init_sim_config(self, seed):
        return gen_rou_file(self.data_path,
                            self.peak_flow1,
                            self.peak_flow2,
                            self.init_density,
                            seed=seed,
                            thread=self.sim_thread)

    def _init_sim(self, seed, gui=False):
        sumocfg_file = self._init_sim_config(seed)
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        #if self.name != 'real_net':
        #    command += ['--time-to-teleport', '600'] # long teleport for safety
        #else:
        command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        command += ['--ignore-route-errors', 'True']
        # collect trip info if necessary
        #if self.is_record:
        command += ['--tripinfo-output',
                    self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        subprocess.Popen(command)
        # wait 2s to establish the traci server
        time.sleep(2)
        self.sim = traci.connect(port=self.port)
        self._simulate(10)


    def terminate(self):
        self.sim.close()

    def step(self, step_info): #step_info #tuple(edge,vehicle,state,action, action_values)
        step_info = self._set_route(step_info)
        reward = []

        self._simulate(10)

        observations = []
        for step_info_instance in step_info:
            edge_name = step_info_instance[0]

            #edge = edges[edge_name]
            reward = self.sim.edge.getLastStepVehicleNumber(edge_name)*-1
            step_info_instance = list(step_info_instance)
            #print(reward)
            step_info_instance.append(reward)
            step_info_instance = tuple(step_info_instance)

            observations.append(step_info_instance)

        done = False

        if self.cur_step >= self.no_of_episode_step:
            done = True



        return observations,done

#actio is 2d list each row has actions for corresponding edges
    def _set_route(self, step_info):
        obs = []

        for step_info_instance in step_info:
            edge = step_info_instance[0]
            veh = step_info_instance[1]
            a = step_info_instance[3]
            #print(step_info_instance[0], a)
            old_route = list(self.sim.vehicle.getRoute(veh))
            current_edge_index = old_route.index(edge)

            if(len(old_route)-current_edge_index>2):

                found_route = list(self.sim.simulation.findRoute(a,old_route[-1]).edges)
                #print(old_route)
                #print(found_route)
                found_route = found_route[1:-1]
                new_route = old_route[0:current_edge_index+1] + found_route
                #print(new_route)
                #print(new_route)
                #new_route = list(new_route.edges)
                #new_route.insert(0,old_route[0])
                #print("old",old_route)
                #print("new", new_route)
                #old_route = tuple(li)
                #old_route[1] = a
                step_info_instance = list(step_info_instance)
                step_info_instance.append(0)
                step_info_instance = tuple(step_info_instance)
                obs.append(step_info_instance)
            else:
                step_info_instance = list(step_info_instance)
                step_info_instance.append(1)
                step_info_instance = tuple(step_info)
                obs.append(step_info_instance)


            try:
                self.sim.vehicle.setRoute(veh,tuple(new_route))
            except:
                print("route assignment failed")
                pass

            return obs






    def _simulate(self, num_step):
        # reward = np.zeros(len(self.control_node_names))
        for _ in range(num_step):
            self.sim.simulationStep()
            # self._measure_state_step()
            # reward += self._measure_reward_step()
            self.cur_step = self.cur_step+1
            #if self.is_record:
                # self._debug_traffic_step()
                #self._measure_traffic_step()
        # return reward
