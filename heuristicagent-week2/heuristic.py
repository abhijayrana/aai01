import gymnasium as gym
import warnings

# Ignore warning messages
warnings.filterwarnings("ignore")

# Set up the environment
env = gym.make('highway-v0', render_mode='rgb_array')

# Configure the environment
env.configure({
    "action": {
        "type": "DiscreteMetaAction",
    },
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-30, 30],
            "y": [-30, 30],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "grid_size": [[-50, 50], [-50, 50]],
        "grid_step": [2, 2],
        "absolute": False
    },
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 100,  # [seconds]
    "initial_spacing": 2,
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 0.25,  # [Hz]
    "render_agent": True,
    "manual_control": False
})

# Reset the environment to start a new simulation
obs, _ = env.reset()

# Function to decide what action to take
def decideMove(env, obs, speed):
    side_safety_factor = 2  # How cautious to be about cars on the sides
    speed_safety_factor = 0  # How cautious to be about speeding up/slowing down

    available_actions = env.get_available_actions()

    # Check for cars on the left
    left_risk_start = 23 - side_safety_factor
    left_risk_end = 35 + side_safety_factor
    left_risk_region = obs[0][23][left_risk_start:left_risk_end]
    left_risk = any(left_risk_region)
    if 0 not in available_actions:
        left_risk = True

    # Check for cars on the right
    right_risk_start = 23 - side_safety_factor
    right_risk_end = 35 + side_safety_factor
    right_risk_region = obs[0][27][right_risk_start:right_risk_end]
    right_risk = any(right_risk_region)
    if 2 not in available_actions:
        right_risk = True

    # Check for cars in front
    front_risk_start = 26
    front_risk_end = 47
    front_region = obs[0][25][front_risk_start:front_risk_end]
    car_in_front = any(front_region)

    # Check if the car in front is moving slowly
    if car_in_front:
        forward_speed_region = [obs[3][25][i] for i in range(front_risk_start, front_risk_end)]
        forward_risk = any([speed < 1 for speed in forward_speed_region])
        pseudo_forward_risk = any([speed < 2 for speed in forward_speed_region])
    else:
        forward_risk = False
        pseudo_forward_risk = False

    # Decide what action to take based on the observations
    if forward_risk:
        if left_risk:
            if right_risk:
                return 4  # Decelerate
            else:
                return 2  # Move right
        elif not right_risk:
            return 0  # Move left
        else:
            return 2  # Move right
    elif pseudo_forward_risk:
        return 4  # Decelerate
    else:
        if speed > 25 - speed_safety_factor:
            return 4  # Decelerate
        elif speed < 22 - speed_safety_factor:
            return 3  # Accelerate
        else:
            return 1  # Idle

done = truncated = False
total_speed = ticks = speed = total_distance = 0
chosen_actions = [0, 0, 0, 0, 0]

# Run the simulation
while not done and not truncated:
    action = decideMove(env, obs, speed)
    chosen_actions[action] += 1
    obs, reward, done, truncated, info = env.step(action)
    speed = info['speed']
    total_distance += reward
    env.render()
    if not done and not truncated:
        total_speed += info['speed']
        ticks += 1

actions = {
    0: 'Left',
    1: 'Idle',
    2: 'Right',
    3: 'Accelerate',
    4: 'Decelerate'
}

average_speed = total_speed / ticks if ticks else 0

# Print the results
print("*********")
print("Results:")
print(f"Average speed: {round(average_speed, 2)}")
print(f"Total distance traveled: {round(total_distance, 2)}")
print("Actions:")
for action, count in enumerate(chosen_actions):
    print("    "+f"{actions[action]}: {count}")
