import seaborn as sns
import matplotlib.pyplot as plt
import os

def heatmap(cars_list, num_of_division, e, save_dir):
    total_of_reward_step_list = [[{"reward":0, "step":0} for i in range(num_of_division)] for j in range(num_of_division)]
    for car in cars_list:
        for i in range(24):
            for key, experience in car.experience[i].items():
                index_x, index_y = key
                reward = experience["reward"]
                step = experience["step"]
                total_of_reward_step_list[index_y][index_x]["reward"] += reward
                total_of_reward_step_list[index_y][index_x]["step"] += step

    evaluation_value_list = [[0 for i in range(num_of_division)] for j in range(num_of_division)]
    for i in range(num_of_division):
        for j in range(num_of_division):
            reward = total_of_reward_step_list[i][j]["reward"]
            step = total_of_reward_step_list[i][j]["step"]
            if step == 0:
                evaluation_value_list[i][j] = 0
            else:
                evaluation_value_list[i][j] = reward / step
    
    plt.figure()
    sns.heatmap(evaluation_value_list, cmap="Blues")
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.savefig(os.path.join(save_dir, str(e) + "Evaluation_value_each_area.png"))
    plt.close("all")

def reward(total_rewards, e, save_dir):
    reward_change = []
    prev_reward = total_rewards[0]
    for i in range(len(total_rewards)):
        if i % 5000 == 0:
            reward_change.append(total_rewards[i] - prev_reward)
            prev_reward = total_rewards[i]
    reward_change.append(total_rewards[-1] - prev_reward)

    plt.figure()
    plt.plot(reward_change)
    plt.xlabel("simulation step (*5000)")
    plt.ylabel("earned reward")
    plt.title("Changes in earned rewards")
    plt.savefig(os.path.join(save_dir, str(e) + "reawrd.png"))
    plt.close("all")