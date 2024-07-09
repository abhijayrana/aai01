import gymnasium as gym
import warnings
import time

warnings.filterwarnings("ignore")

def run_random_agent_simulation():
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.configure({
        "duration": 100,
        "lanes_count": 5,
        "screen_width": 1200,
        "screen_height": 800,
    })
    env.reset()


    total_distance = 0
    total_collisions = 0
    total_speed = 0
    total_collision_free_speed = 0
    collision_free_ticks = 0
    chosen_actions = [0, 0, 0, 0, 0]
    ticks_counted = 0


    truncated = done = False
    while not (done or truncated):
        action_taken = env.action_space.sample()

        chosen_actions[action_taken] += 1
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


    average_speed = total_speed / ticks_counted if ticks_counted else 0
    average_collision_free_speed = total_collision_free_speed / collision_free_ticks if collision_free_ticks else 0
    distance_km = total_distance / 1000
    collisions_per_1000m = total_collisions / distance_km if distance_km else 0

    return {
        "average_speed": average_speed,
        "total_collisions": total_collisions,
        "collisions_per_1000m": collisions_per_1000m,
        "average_collision_free_speed": average_collision_free_speed,
        "actions_distribution": chosen_actions,
    }

def aggregate_results(results):
    total_tests = len(results)
    aggregated_data = {
        "average_speed": sum(res["average_speed"] for res in results) / total_tests,
        "total_collisions": sum(res["total_collisions"] for res in results),
        "collisions_per_1000m": sum(res["collisions_per_1000m"] for res in results) / total_tests,
        "average_collision_free_speed": sum(res["average_collision_free_speed"] for res in results) / total_tests,
        "actions_distribution": [0, 0, 0, 0, 0],
    }

    for res in results:
        for i in range(5):
            aggregated_data["actions_distribution"][i] += res["actions_distribution"][i]

    aggregated_data["actions_distribution"] = [count / total_tests for count in aggregated_data["actions_distribution"]]

    return aggregated_data

# Run multiple tests and aggregate the data
num_tests = 10
results = [run_random_agent_simulation() for _ in range(num_tests)]
aggregated_data = aggregate_results(results)


print("____________________")
print("AGGREGATED RESULTS AFTER 10 TESTS")
print()
print(f"Average speed: {round(aggregated_data['average_speed'], 2)} km/h")
print(f"Total collisions: {aggregated_data['total_collisions']}")
print(f"Collisions per 1000 meters: {round(aggregated_data['collisions_per_1000m'], 2)}")
print(f"Average collision-free speed: {round(aggregated_data['average_collision_free_speed'], 2)} km/h")
print(f"Actions distribution: {aggregated_data['actions_distribution']}")
print()
