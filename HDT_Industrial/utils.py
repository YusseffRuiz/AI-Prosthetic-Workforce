import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.signal import find_peaks
from scipy import stats


def evaluate(model=None, env=None, algorithm="random", num_episodes=5, no_done=False, max_episode_steps=1000, render=False):
    total_rewards = []
    range_episodes = num_episodes
    mujoco_env = hasattr(env, "sim")
    muscle_flag = hasattr(env, "muscles_enable")
    min_distance = 5
    successful_episodes = 0
    max_attempts_per_episode = 50  # Maximum retries per episode
    energy_episodes = []
    distance_episodes = []
    phases_hip_tot = []
    phases_ankle_tot = []
    while successful_episodes < num_episodes:
        attempts = 0
        success = False
        while not success and attempts < max_attempts_per_episode:
            if mujoco_env:
                obs = env.reset()
            else:
                obs = env.reset()[0]
            if muscle_flag:
                muscle_states = env.muscle_states
            done = False
            episode_reward = 0
            cnt = 0
            distance = 0
            energy = 0
            phases_hip = []
            phases_ankle = []

            while not done:
                cnt += 1
                with torch.no_grad():
                    if algorithm != "random":
                        # if algorithm == "MPO":
                        #     action = model(obs)
                        if muscle_flag:
                            action = model.test_step(observations=obs, muscle_states=muscle_states, steps=successful_episodes)
                        else:
                            action = model.test_step(observations=obs, steps=successful_episodes)
                    else:
                        action = env.action_space.sample()
                if len(action.shape) > 1:
                    action = action[0, :]
                obs, reward, done, info, extras = env.step(action)

                # Get values for plotting
                public_obs = env.unwrapped.public_joints()
                phases_hip.append(public_obs[0, 1])  # 0,1 is right Hip
                phases_ankle.append(public_obs[1, 1])  # 1,1 is right Ankle

                distance = extras["distance_from_origin"]
                energy += np.sum(extras["step_energy"])
                if muscle_flag:
                    muscle_states = env.muscle_states
                if render:
                    env.sim.renderer.render_to_window()
                episode_reward += reward
                if no_done:
                    done = False
                if cnt >= max_episode_steps:
                    done = True
                    # Check success criteria
            if distance >= min_distance:  # Successful episode
                success = True
                successful_episodes += 1
                energy_episodes.append(energy)
                distance_episodes.append(distance)
                total_rewards.append(episode_reward)
                phases_hip_tot.append(np.array(phases_hip))
                phases_ankle_tot.append(np.array(phases_ankle))
                print(
                    f"Episode {successful_episodes}/{range_episodes}: Reward = {episode_reward}. "
                    f"Distance: {distance} mts, Energy: {energy}")
    average_reward = np.mean(total_rewards)
    env.close()
    print(f"Average Reward over {range_episodes} episodes: {average_reward}")
    min_length = min(len(arr) for arr in phases_hip_tot)

    # Trim all arrays to the minimum length
    phases_hip_tot = [arr[:min_length] for arr in phases_hip_tot]

    return np.array(distance_episodes), np.array(energy_episodes), np.array(get_energy_per_meter(np.array(energy_episodes), np.array(distance_episodes))), np.array(phases_hip_tot)


def retrieve_cpg(config):
    # Loading CPG configuration
    cpg_oscillators = config.cpg_oscillators
    cpg_neurons = config.cpg_neurons
    cpg_tau_r = config.cpg_tau_r
    cpg_tau_a = config.cpg_tau_a

    return cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a


def load_mpo(path, environment, checkpoint="last"):
    config, checkpoint_path = load_config_and_paths(path, checkpoint)
    header = config["tonic"]["header"]
    agent = config["tonic"]["agent"]
    # Run the header
    exec(header)
    # Build the agent.
    agent = eval(agent)
    # Adapt mpo specific settings
    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
    )
    # Load the weights of the agent form a checkpoint.
    agent.load(checkpoint_path, only_checkpoint=True)
    return agent

def load_config_and_paths(checkpoint_path, checkpoint="last"):
    if checkpoint_path.split("/")[-1] != "checkpoints":
        checkpoint_path += "checkpoints"
    if not os.path.isdir(checkpoint_path):
        return None, None, None

    # List all the checkpoints.
    checkpoint_ids = []
    for file in os.listdir(checkpoint_path):
        if file[:5] == "step_":
            checkpoint_id = file.split(".")[0]
            checkpoint_ids.append(int(checkpoint_id[5:]))

    if checkpoint_ids:
        # Use the last checkpoint.
        if checkpoint == "last":
            checkpoint_id = max(checkpoint_ids)
            checkpoint_path = os.path.join(
                checkpoint_path, f"step_{checkpoint_id}"
            )

        # Use the specified checkpoint.
        else:
            checkpoint_id = int(checkpoint)
            if checkpoint_id in checkpoint_ids:
                checkpoint_path = os.path.join(
                    checkpoint_path, f"step_{checkpoint_id}"
                )
            else:
                checkpoint_path = None

    else:
        checkpoint_path = None

    # Load the experiment configuration.
    arguments_path = os.path.join(
        checkpoint_path.split("checkpoints")[0], "config.yaml"
    )
    with open(arguments_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config, checkpoint_path


def plot_data(data, plot_name='data', units='', save=False):
    # Create a bar chart
    total_values = [np.mean(dat) for dat in data]
    std_values = [np.std(dat) for dat in data]


    plt.figure(figsize=(8, 5))
    bars = plt.bar(["Full-Abled", "Handicap"], total_values, yerr=std_values, capsize=5, color=plt.colormaps.get_cmap("tab20").colors,
                   label="Mean ± Std")

    # Add precise value labels on top of each bar
    for bar, val in zip(bars, total_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha='center', va='bottom', fontsize=10)
    plt.xlabel('Algorithm')
    plt.ylabel(f'Total {plot_name} ({units})')
    plt.title(f'Total {plot_name} Comparison')

    if save:
        plt.savefig(f"../HDT_Images/{plot_name}_plot.png")
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()

def get_usage_percentage(full_able_data, handicapped_data, data_name=""):
    percentage_increase = (handicapped_data - full_able_data) / full_able_data * 100
    if percentage_increase < 0:
        sign = "less"
    else:
        sign = "more"
    print(f"Handicaped Worker show {abs(percentage_increase):.2f}% {sign} {data_name} than the Full Abled Worker")


def get_energy_per_meter(total_energy, total_distance, save_folder=None, plot_fig=False, x_range=(0,40), norm=False):
    # Calculate energy per meter for each episode
    energy_per_meter = total_energy / total_distance  # This gives the energy per meter for each episode
    # Normalize values for better visualization, future deployment
    energy_per_meter_normalized = (energy_per_meter - np.min(energy_per_meter)) / (
            np.max(energy_per_meter) - np.min(energy_per_meter)
    )
    distance_normalized = (total_distance - np.min(total_distance)) / (
            np.max(total_distance) - np.min(total_distance)
    )

    return energy_per_meter


def get_motion_pattern(data):
    autocorr = np.correlate(data, data, mode='full')
    # autocorr = correlate(data, data, mode='full')  # Compute autocorrelation
    autocorr = autocorr[len(autocorr) // 2:]  # Take positive lags

    # Find peaks in autocorrelation to detect cycles
    peaks, _ = find_peaks(autocorr, height=0.1 * max(autocorr), distance=10)
    if len(peaks) > 1:
        cycle_length = peaks[1] - peaks[0]  # Approximate cycle length
    else:
        cycle_length = len(data) // 5  # Default fallback

    if cycle_length < 5:
        print("No clear pattern detected.")
        return

    pattern_segment = data[:cycle_length]  # Extract first cycle

    # Count how many times the cycle repeats in the full dataset
    step_count = len(data) // cycle_length

    return cycle_length, peaks, pattern_segment, step_count

# Process multiple episodes and extract aligned motion cycles
def process_multiple_episodes(motion_episodes, joint="Joint", num_episodes=5, worker="worker", distance=0):
    """
    Process hip motion data across multiple episodes and extract aligned cycles.

    Returns:
    - all_patterns: Aligned motion cycles from all episodes.
    - mean_pattern: Average motion cycle across all episodes.
    - std_pattern: Standard deviation of motion cycles.
    """
    all_patterns = []

    steps_mean = 0
    for episode_idx, motion in enumerate(motion_episodes):
        cycle_length, peaks, pattern_segment, step_count = get_motion_pattern(motion)
        steps_mean += step_count

        if cycle_length < 5:
            print(f"Episode {episode_idx}: No clear pattern detected.")
            continue

        # Extract motion cycles based on detected peaks
        extracted_cycles = []
        for i in range(len(peaks) - 1):
            cycle_start, cycle_end = peaks[i], peaks[i] + int(cycle_length)
            if cycle_end <= len(motion):
                extracted_cycles.append(motion[cycle_start:cycle_end])

        if extracted_cycles:
            all_patterns.append(np.mean(extracted_cycles, axis=0))  # Compute mean per episode
    steps_mean /= num_episodes

    print(f"{worker} Worker gave a total of {steps_mean:.2f} steps mean across all episodes over {distance:.2f} meters.")



    return all_patterns, steps_mean

def compare_steps(steps_mean, distance_mean):
    # [0] is for full abled, [1] is for hadicaped
    steps_per_meter_fa = steps_mean[0]/distance_mean[0]
    steps_per_meter_hc = steps_mean[1]/distance_mean[1]

    print(f"Full Able worker gave {steps_per_meter_fa} per meter, while Handicapped worker gave {steps_per_meter_hc} per meter.")

    diff = steps_per_meter_hc - steps_per_meter_fa

    if diff < 0:
        sign = "less"
    else:
        sign = "more"
    print(f"Handicapped Worker gave {abs(diff):.2f} {sign} steps than the Full Abled Worker")