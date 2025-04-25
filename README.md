# Cars are fast :truck:

This project aims to train Reinforcement Learning (RL) agents in autonomous driving environments provided by the `highway-env` library.

<p align="center">
  <img src="images/dejavu.gif" width="400"/>
</p>

Three environments are used divided in the repository structure:
- `car_lane`: DQN agent on the `Highway` environment
- `race_track`: SAC+DDPG agents on the `RaceTrack` environment
- `roundabout`: PPO+DQN agents and Task4 experiment on the `Roundabout` environment

Task 4 refers to test how the agent trained on the `Roundabout` environment generalize when used in a different environment like `Highway`.

Each subfolder has 3 main scripts:
- script to generate the environment configuration
- script to train the agent
- script to test the agent

These have to be run in the correct order to provide the configuration to the environment used and then test the agent with the trained weights.

In details:
```bash
car_lane
   |-- config_carlane.py #Generate config for highway
   |-- dqn.py #Train the DQN agent
   |-- test_dqn.py #Inference of the trained agent
race_track
   |-- config_racetrack.py #Generate config for racetrack
   |-- ddpg.py #Train and test DDPG
   |-- sac.py #Train and test SAC
roundabount
   |-- config_highway_task4.py #Generate config for task 4
   |-- config_roundabout.py #Generate config for roundabout
   |-- dqn.py #Train DQN
   |-- ppo.py #Train PPO
   |-- test_highway_task4.py #Infer the task 4
   |-- test_roundabout_dqn.py #Infer the DQN agent on roundabout
   |-- test_roundabout_ppo.py #Infer the PPO agent on roundabout
```