import os
import matplotlib.pyplot as plt

path = "./reward/"
files = []

for x in os.listdir(path):
    if os.path.isfile(path + x):
        files.append(x)
files.sort()

# output bar
plt.figure()
total_reawrd = []
left = [1,2,3,4,5,6,7,8]
label = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
for infilename in files:
    with open(path + infilename, "r") as f:
        for line in f:
            reward = float(line.replace("\n", ""))
    total_reawrd.append(reward)
plt.bar(left, total_reawrd, tick_label=label)
plt.xlabel("epsilon")
plt.ylabel("total rewards")
plt.title("total rewards")
plt.savefig("rewards_bar.png")
plt.close("all")

# output total reward
plt.figure()
for infilename in files:
    e = infilename[13:-4]
    total_rewards = []
    with open(path + infilename, "r") as f:
        for line in f:
            reward = float(line.replace("\n", ""))
            total_rewards.append(reward)
    plt.plot(total_rewards, label="e_" + str(e))
plt.legend(loc="lower right")
plt.xlabel("simulation steps")
plt.ylabel("total rewards")
plt.title("Changes in earned rewards")
plt.savefig("total_reawrds.png")
plt.close("all")


# output reward per 5000step
plt.figure()
for infilename in files:
    e = infilename[13:-4]
    total_rewards = []
    with open(path + infilename, "r") as f:
        for line in f:
            reward = float(line.replace("\n", ""))
            total_rewards.append(reward)
    
    reward_change = []
    prev_reward = total_rewards[0]
    for i in range(len(total_rewards)):
        if i % 5000 == 0:
            reward_change.append(total_rewards[i] - prev_reward)
            prev_reward = total_rewards[i]
    reward_change.append(total_rewards[-1] - prev_reward)

    plt.plot(reward_change, label="e_" + str(e))

plt.legend(loc="lower right")
plt.xlabel("simulation steps (*5000)")
plt.ylabel("earned reward")
plt.title("reward per every 5000 steps")
plt.savefig("reawrds_per_5000step.png")
plt.close("all")