import numpy as np

import Adaptive_RL
import utils
import myosuite
import deprl

"""
TODO:

- Set Goals
- Run the three environments in a single loop.
- Statistical Test for X meters.

"""


def main_test():

    num_episodes = 50
    sim_time = 600
    # we can pass arguments to the environments here
    env_handi = "myoAmp1DoFWalk-v0"
    env_myoleg = "myoLegWalk"

    dir_myoLeg = "Testing_Folder/myoLeg/logs/"
    dir_SAC = "Testing_Folder/myoAmp1DoFWalk-v0-SAC-CPG/logs/"

   # Preparation for Full Able Worker Environment
    env_abled = Adaptive_RL.MyoSuite(env_myoleg, reset_type='random', scaled_actions=False)
    env_abled = Adaptive_RL.apply_wrapper(env_abled, direct=True)
    # Loading of Full abled worker agent.
    path, config, _ = Adaptive_RL.get_last_checkpoint(path=dir_myoLeg)
    agent_abled = utils.load_mpo(dir_myoLeg, env_abled)
    print("Loaded weights for {} Worker, path: {}".format("Full Abled", path))


    # Loading of Prosthetic Worker agent
    env_handicap = Adaptive_RL.MyoSuite(env_handi, reset_type='random', scaled_actions=False)
    env_handicap = Adaptive_RL.apply_wrapper(env_handicap, direct=True)

    path, config, _ = Adaptive_RL.get_last_checkpoint(path=dir_SAC)
    cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a = utils.retrieve_cpg(config)
    env_handicap = Adaptive_RL.wrap_cpg(env_handicap, env_handi, cpg_oscillators, cpg_neurons, cpg_tau_r, cpg_tau_a)
    agent_handicap, _ = Adaptive_RL.load_agent(config, path, env_handicap, muscle_flag=True)
    print("Loaded weights for {} Worker, path: {}".format("Handicap", path))




    distance_fullabled, energy_fullAbled, energy_per_meter_fullAbled, hip_motion_fullAbled = utils.evaluate(agent_abled, env=env_abled, algorithm="MPO", num_episodes=num_episodes, max_episode_steps=sim_time,
                    no_done=False)

    distance_handicap, energy_handicap, energy_per_meter_handicap, hip_motion_handicap = utils.evaluate(agent_handicap, env=env_handicap, algorithm="SAC", num_episodes=num_episodes, max_episode_steps=sim_time,
                   no_done=False)

    dataDistance = [distance_fullabled, distance_handicap]
    dataEnergy = [energy_fullAbled, energy_handicap]
    dataEnergyPerMeter = [energy_per_meter_fullAbled, energy_per_meter_handicap]

    utils.plot_data(dataDistance, plot_name="Distance", units="Mts", save=True)
    utils.plot_data(dataEnergy, plot_name="Energy", units="Joules", save=True)
    utils.plot_data(dataEnergyPerMeter, plot_name="Energy per Meter", units="Joules/m", save=True)

    dataDistance = [np.mean(dat) for dat in dataDistance]
    dataEnergy = [np.mean(dat) for dat in dataEnergy]
    dataEnergyPerMeter = [np.mean(dat) for dat in dataEnergyPerMeter]

    utils.get_usage_percentage(dataDistance[0], dataDistance[1], data_name="Distance travelled")
    utils.get_usage_percentage(dataEnergy[0], dataEnergy[1], data_name="Energy usage")
    utils.get_usage_percentage(dataEnergyPerMeter[0], dataEnergyPerMeter[1], data_name="Energy per Meter")

    steps_mean = [0, 0]
    _, steps_mean[0] = utils.process_multiple_episodes(hip_motion_fullAbled, joint="hip motion", num_episodes=num_episodes,
                                    worker="Full Abled", distance=dataDistance[0])
    _, steps_mean[1] = utils.process_multiple_episodes(hip_motion_fullAbled, joint="hip motion", num_episodes=num_episodes,
                                    worker="Handicap Worker", distance=dataDistance[1])

    utils.compare_steps(steps_mean, dataDistance)


    print("Tests Finished")

if __name__ == '__main__':
    main_test()


"""
Handicaped Worker show 7.55% less Distance travelled than the Full Abled Worker
Handicaped Worker show 22.17% more Energy usage than the Full Abled Worker
Handicaped Worker show 32.60% more Energy per Meter than the Full Abled Worker
Full Abled Worker gave a total of 5.76 steps mean across all episodes over 7.75 meters.
Handicap Worker Worker gave a total of 5.76 steps mean across all episodes over 7.16 meters.
Full Able worker gave 0.7433961093794658 per meter, while Handicapped worker gave 0.8040796226820717 per meter.
Handicapped Worker gave 0.06 more steps than the Full Abled Worker
"""