import pickle

config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 128),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    # "observation": {
    #         "type": "OccupancyGrid",
    #         "vehicles_count": 5,
    #         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    #         "features_range": {
    #             "x": [-100, 100],
    #             "y": [-100, 100],
    #             "vx": [-20, 20],
    #             "vy": [-20, 20]
    #         },
    #         "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    #         "grid_step": [5, 5],
    #         "absolute": False
    #     },
    "action": {"type": "DiscreteMetaAction"},
    "collision_reward": -4,
    "high_speed_reward": 0,
    "right_lane_reward": 0,
    "lane_change_reward": 0,
    "incoming_vehicle_destination": None,
    "duration": 11,  # [s] If the environment runs for 11 seconds and still hasn't done(vehicle is crashed), it will be truncated. "Second" is expressed as the variable "time", equal to "the number of calls to the step method" / policy_frequency.
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px] width of the pygame window
    "screen_height": 600,  # [px] height of the pygame window
    "centering_position": [
        0.5,
        0.6,
    ],  # The smaller the value, the more southeast the displayed area is. K key and M key can change centering_position[0].
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "normalize_reward": False,
}


with open("config.pkl", "wb") as f:
    pickle.dump(config, f)
