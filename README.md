# Let's drive until we die

This project aims to train Reinforcement Learning (RL) agents in autonomous driving environments provided by the `highway-env` library.

Three environments are used divided in the repository structure:
- `car_lane`: DQN agent on the `Highway` environment
- `race_track`: SAC agent on the `RaceTrack` environment
- `roundabout`: PPO agent on the `Roundabout` environment

Each subfolder has 3 main scripts:
- script to generate the environment configuration
- script to train the agent
- script to test the agent

These have to be run in the correct order to provide the configuration to the environment used and then test the agent with the trained weights.