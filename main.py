# Main function of the RL air combat training programe
#
# Hasan ISCI - 27.12.2020

import gym
import gym_dubins_airplane
from matplotlib import pyplot as plt
import numpy as np

DebugInfo = False
RenderSteps = True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # To test directly aircraft model
    # simple_ac = ACEnvironment2D()
    # simple_ac.takeaction(0.,0.,5.)

    env = gym.make('dubinsAC2D-v0', actions='cont')
    state = env.reset()

    reward_history=[]

    # Run environment with arbitrary actions for testing purposes
    for _ in range(10000):

        action = [0, 0, 0.1]

        if state[5] > 20:
            action[0] = 0.2
        elif state[5] < -20:
            action[0] = -0.2
        else:
            action[0] = 0.

        if state[6] > 20:
            action[1] = 0.2
        elif state[6] < -20:
            action[1] = -0.2
        else:
            action[1] = 0.

        state, reward, terminate, info = env.step(action)  # take a random action

        reward_history.append(reward)

        if RenderSteps:
            env.render()
        if DebugInfo:
            print({'obs': state,
                   'reward': reward,
                   'terminate': terminate,
                   'info': info})
        if terminate:

            plt.plot( np.array(reward_history) )
            plt.show()

            reward_history.clear()

            state = env.reset()

    env.close()

