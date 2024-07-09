import gymnasium as gym
import warnings
import keyboard
import time

warnings.filterwarnings("ignore")

env = gym.make('highway-v0', render_mode='rgb_array')
env.configure({
    "duration": 100,
    "lanes_count": 5,
    "manual_control": True,
    "screen_width": 1200, 
    "screen_height": 800,
})
env.reset()


total_distance = 0
total_collisions = 0
total_speed = 0
total_collision_free_speed = 0
collision_free_ticks = 0
actions_taken = [0, 0, 0, 0, 0]
ticks_counted = 0


ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}


truncated = done = False
while not (done or truncated):
    action_taken = 1  # Default action: IDLE

    if keyboard.is_pressed('up'):
        action_taken = 0  # LANE_LEFT
    elif keyboard.is_pressed('down'):
        action_taken = 2  # LANE_RIGHT
    elif keyboard.is_pressed('right'):
        action_taken = 3  # FASTER
    elif keyboard.is_pressed('left'):
        action_taken = 4  # SLOWER

    actions_taken[action_taken] += 1
    obs, reward, done, truncated, info = env.step(action_taken)
    env.render()

    speed = info['speed']
    total_speed += speed
    ticks_counted += 1
    if not info.get('crashed', False):
        total_collision_free_speed += speed
        collision_free_ticks += 1
    else:
        total_collisions += 1

    total_distance += speed / env.config["simulation_frequency"]


env.close()

# Calculate and print stats
average_speed = total_speed / ticks_counted if ticks_counted else 0
average_collision_free_speed = total_collision_free_speed / collision_free_ticks if collision_free_ticks else 0
distance_km = total_distance / 1000
collisions_per_1000m = total_collisions / distance_km if distance_km else 0

print("____________________")
print("FINISHED SIMULATION.")
print()
print(f"Average speed: {round(average_speed, 2)} km/h")
print(f"Total collisions: {total_collisions}")
print(f"Collisions per 1000 meters: {round(collisions_per_1000m, 2)}")
print(f"Average collision-free speed: {round(average_collision_free_speed, 2)} km/h")
print(f"Actions distribution: {actions_taken}")
print()
