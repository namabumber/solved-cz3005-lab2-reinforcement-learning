Download Link: https://assignmentchef.com/product/solved-cz3005-lab2-reinforcement-learning
<br>



In this project, you need to implement one reinforcement learning algorithm (e.g., value iteration, policy iteration, Q-learning) for one grid-world-based environment: Treasure Hunting.

(a) 3D grid world. Smile faces represent terminal states which (b) The illustration of transition, e.g., the ingive reward 1.          tended action is RIGHT

Figure 1: Illustration of treasure hunting in a cube

<h1>2           Treasure Hunting in a Cube</h1>

The environment is a 3D grid world. The MDP formulation is described as follows:

<ul>

 <li>State: a 3D coordinate, which indicates the current position where the agent is. The initial state is (0, 0, 0) and there is only one terminal state: (3,3,3).</li>

 <li>Action: The action space is (forward, backward, left, right, up, down). The agent needs to select one of them to navigate in the environment.</li>

 <li>Reward: The agent will receive 1 reward when it arrives at the terminal states, or otherwise receive -0.1 reward.</li>

 <li>Transition: The intended movement happens with probability 0.6. With probability 0.1, the agent ends up in one of the states perpendicular to the intended direction. If a collision with a wall happens, the agent stays in the same state.</li>

</ul>

<h1>3           Code Example</h1>

We provide the environment code environment.py and examples code test.py. In environment.py, we provide the code: TreasureCube.

In test.py, we provide a random agent. You can modify it to implement your agent. You should install a numpy package additionally to run the code.

<table width="624">

 <tbody>

  <tr>

   <td width="624">from collections import defaultdict import argparse import random import numpy as np from environment import TreasureCube# you need to implement your agent based on one RL algorithm class RandomAgent(object):def __init__(self):self.action_space = [’left’,’right’,’forward’,’backward’,’up’,’down’] # inTreasureCube self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))def take_action(self, state):action = random.choice(self.action_space) return action# implement your train/update function to update self.V or self.Q# you should pass arguments to the train function def train(self, state, action, next_state, reward):pass</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

Besides, in test.py, we implement a test function. You should replace the random agent with your agent in line 3.

<table width="624">

 <tbody>

  <tr>

   <td width="23">def</td>

   <td width="601">test_corridor(max_episode, max_step):env = TreasureCorridor(max_step=max_step) agent = RandomAgent()for epsisode_num in range(0, max_episode):state = env.reset() terminate = Falset = 0episode_reward = 0 while not terminate:action = agent.take_action(state)reward, terminate, next_state = env.step(action) episode_reward += reward# env.render()# print(f’step: {t}, action: {action}, reward: {reward}’) t += 1agent.train(state, action, next_state, reward) state = next_stateprint(f’epsisode: {epsisode_num}, total_steps: {t} episode reward: {episode_reward}’)</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

If you use Q-learning, you can use the parameters: discount factor <em>γ </em>= 0<em>.</em>99, learning rate <em>α </em>= 0<em>.</em>5, exploration rate

You can run the following code to generate output and test your agent.

<table width="624">

 <tbody>

  <tr>

   <td width="624">python test.py –max_episode 500 –max_step 500</td>

  </tr>

 </tbody>

</table>

1