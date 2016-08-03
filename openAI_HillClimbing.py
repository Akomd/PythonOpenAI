import gym
import numpy as np
env = gym.make('CartPole-v0')
maxGene = 2000
maxSim = 100
M = 5
T = 200
alpha = 0.1

sucNum = 0
for sim in range(maxSim):
    print("Simulation {}".format(sim))
    theta = 2 * np.random.rand(4, 1) - 10
    r_best = 0
    for g in range(maxGene):
        theta_new = theta + alpha * (2 * np.random.rand(4,1) - 1)
        r_total = 0
        for i_episode in range(M):
            observation = env.reset()
            for t in range(T):
                # env.render()
                action = 0 if observation.dot(theta_new) < 0 else 1
                observation, reward, done, info = env.step(action)
                r_total += reward
                if done:
                    # print("Episode finished after {} timesteps".format(t+1))
                    break
        # print("{}  ".format(g) + "{}  ".format(theta_new))
        r_best, theta = (r_total, theta_new) if r_total > r_best else (r_best, theta)
        if r_total >= T * M:
            print("SUCCESS theta = \n{} ".format(theta))
            sucNum += 1
            break
print("SimulationNum = {}".format(maxSim))
print("SuccessNum = {}".format(sucNum))

