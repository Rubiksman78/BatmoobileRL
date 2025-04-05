import pickle

config = {
    "observation": {
        "type": "Kinematics",
        "absolute": True,
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-15, 15],
            "vy": [-15, 15],
        },
    },
    "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
    "incoming_vehicle_destination": None,
    "duration": 20,  # [s] If the environment runs for 11 seconds and still hasn't done(vehicle is crashed), it will be truncated. "Second" is expressed as the variable "time", equal to "the number of calls to the step method" / policy_frequency.
    "collision_reward": -1,
    "high_speed_reward": 0.6,
    "lane_change_reward": -0.05,
    "right_lane_reward": 0,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px] width of the pygame window
    "screen_height": 600,  # [px] height of the pygame window
    # "centering_position": [0.5, 0.6],  # The smaller the value, the more southeast the displayed area is. K key and M key can change centering_position[0].
    # "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "normalize_reward": True,
}

with open("config.pkl", "wb") as f:
    pickle.dump(config, f)
