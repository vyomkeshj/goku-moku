import numpy as np
from stable_baselines3 import SAC, A2C
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from env.gomoku_env import GomokuEnv
from env.utils import BOARD_SIZE

# from policy.custom_cnn import custom_cnn_args
# from wandb.integration.sb3 import WandbCallback
# import wandb

# Train model and test it
if __name__ == '__main__':
    # wandb.init()
    env = GomokuEnv()

    # model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise)
    model = A2C("MlpPolicy", env, verbose=1)
    # model = DDPG.load("ddpg_mountain", action_noise=action_noise)

    model.learn(total_timesteps=18000)
    model.save("discrete-life-015")

    obs = env.reset()
    print("________________TEST_GAME________________")
    for i in range(BOARD_SIZE*BOARD_SIZE):
        action, in_st = model.predict(obs.astype(np.float32), deterministic=False)
        obs, reward, done, info = env.step(action)
        if done:
            env.render()
            break
