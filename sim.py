import os, sys

if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo-gui"
print(sumoBinary)
sumoCmd = [sumoBinary, "-c", "./sumo_iisc.sumocfg"]

import traci
from traci.domain import Domain as DM

traci.start(sumoCmd)
EdgeDomain = traci.edge.getIDList()

print(EdgeDomain)
step = 0
while step < 1000:
   traci.simulationStep()
#   if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
#       traci.trafficlight.setRedYellowGreenState("0", "GrGr")
   step += 1

traci.close()

