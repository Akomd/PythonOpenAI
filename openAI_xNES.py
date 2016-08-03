import gym
import numpy as np
import scipy.linalg as sc

env = gym.make('CartPole-v0')
dim = 4
maxGene = 2000
maxSim = 100
M = 1
T = 200
alpha = 0.1

EYE = np.eye(dim)

sucNum = 0
for sim in range(maxSim):
    isBreak = False
    print("Simulation {}".format(sim))
    theta = 2 * np.random.rand(dim, 1) - 10
    lambda_xNES = 8
    sigma_xNES = 0.5
    B_xNES = EYE
    w_i_xNES = np.zeros(lambda_xNES)
    for i in range(lambda_xNES):
        w_i_xNES[i] = max(0, np.log(lambda_xNES / 2.0 + 1.0) - np.log(i + 1))
    for i in range(lambda_xNES):
        sum_w = np.sum(w_i_xNES)
        w_i_xNES[i] = w_i_xNES[i] / sum_w - 1.0 / lambda_xNES
    eta_m_xNES = 1.0
    eta_sigma_xNES = (3.0 / 5.0) * ((3 + np.log(dim)) / (dim * np.sqrt(dim)))
    eta_B_xNES = eta_sigma_xNES
    r_total = np.zeros(lambda_xNES)
    r_best = 0
    for g in range(maxGene):
        z = np.random.randn(dim, lambda_xNES)
        theta_calc = theta
        for i in range(lambda_xNES - 1):
            theta_calc = np.hstack((theta_calc, theta))
        thetas = theta_calc + sigma_xNES * B_xNES.dot(z)
        for i in range(lambda_xNES):
            theta_new = thetas[0:dim, i:i + 1]
            r_total[i] = 0
            for i_episode in range(M):
                observation = env.reset()
                for t in range(T):
                    # env.render()
                    action = 0 if observation.dot(theta_new) < 0 else 1
                    observation, reward, done, info = env.step(action)
                    r_total[i] += reward
                    if done:
                        break
            if r_total[i] >= T * M:
                print("SUCCESS theta = \n{} ".format(theta_new))
                isBreak = True
        if isBreak:
            sucNum += 1
            break
        sort_ind = np.argsort(r_total)
        z_sort = (np.transpose(np.transpose(z)[sort_ind]))[:, ::-1]
        G_m_xNES = sigma_xNES * B_xNES.dot(np.transpose(np.asmatrix(np.sum(w_i_xNES * z_sort, 1))))
        G_M_xNES = np.zeros((dim, dim))
        for i in range(lambda_xNES):
            G_M_xNES += w_i_xNES[i] * ((z_sort[0:dim, i:i + 1]).dot(np.transpose(z_sort[0:dim, i:i + 1])) - EYE)
        G_sigma_xNES = np.trace(G_M_xNES) / dim
        G_B_xNES = G_M_xNES - G_sigma_xNES * EYE
        theta += eta_m_xNES * G_m_xNES
        sigma_xNES = sigma_xNES * np.exp(eta_sigma_xNES / 2.0 * G_sigma_xNES)
        B_xNES = B_xNES.dot(sc.expm(eta_B_xNES / 2.0 * G_B_xNES))
print("SimulationNum = {}".format(maxSim))
print("SuccessNum = {}".format(sucNum))
