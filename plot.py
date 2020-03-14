import matplotlib.pyplot as plt

accumulated_reward_list = [-19243, -17120, -16482, -17346, -16649, -16293, -18383, -16816, -16813, -16462]


plt.plot(range(1,len(accumulated_reward_list)+1),accumulated_reward_list)
plt.ylabel('reward')
plt.xlabel("episode")
plt.savefig("epoch0episode9.png")
