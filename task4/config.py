import pickle

config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [8, 8],
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "collision_reward": -100,
    "right_lane_reward": 0.5,
    "high_speed_reward": 0.1,
    "reward_speed_range": [20, 30],
    "merging_speed_reward": -0.5,
    "lane_change_reward": 0,
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}
with open("config.pkl", "wb") as f:
    pickle.dump(config_dict, f)
