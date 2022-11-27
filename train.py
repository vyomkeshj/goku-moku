import time

import numpy as np
import torch

from env.gomoku_env import fast_environment
from policy.TD3_agent import TD3
from policy.replay_buffer import ReplayBuffer

if __name__ == "__main__":

    policy = "TD3"
    seed = 0
    start_timesteps = 30e3
    eval_freq = 5e3
    max_timesteps = 40e5
    expl_noise = 0.1
    batch_size = 128
    discount = 0.99
    tau = 0.0005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    env = fast_environment()
    # env = java_environment()

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
    }

    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        policy = TD3(**kwargs)
        # policy.load("/content/drive/My Drive/Colab Notebooks/models/14_orpos_reweighted/model-fastrobot")

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, env, seed)]

    state, done = env.reset(), False
    starting_point = env.robot_mdp.get_robot_init_tcp()

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    start_time = time.time()
    f_start_time = time.time()

    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action\
        next_state, reward, done, _ = env.step(action)

        if not bool(done):
            dist_array = np.append(dist_array, [env.robot_mdp.get_distance_from_target()], axis=0)
            ang_array = np.append(ang_array, [abs(env.robot_mdp.get_current_rot_difference())], axis=0)
            rew_buf = np.append(rew_buf, np.array([reward]), axis=0)

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        epoch_end = False if episode_timesteps < env._max_episode_steps else True
        # Store data in replay buffer,
        # todo do the HER modification here
        # print("adding state to replay buffer", state)

        replay_buffer.add(state, action, next_state, reward, done_bool)  # goes to temp buffer

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if bool(done) | epoch_end:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print("Total T:", {t + 1}, " Episode Num:", {episode_num + 1}, " Episode T:", {episode_timesteps},
                  " Reward:", {episode_reward}, "--- time = %s seconds ---" % (time.time() - start_time))
            target_stored = False

            # Reset environment
            start_time = time.time()
            state, done = env.reset(), False
            starting_point = env.robot_mdp.get_robot_init_tcp()

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            policy.save("./model-moku")
    print("time taken= ", time.time() - f_start_time)