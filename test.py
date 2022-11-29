import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from env.gomoku_env import GomokuEnv

# from policy.custom_cnn import custom_cnn_args
# from wandb.integration.sb3 import WandbCallback
# import wandb

# Train model and test it
if __name__ == '__main__':
    # wandb.init()
    env = GomokuEnv()
    param_noise = None
    model = SAC.load("sac-one")

    obs = env.reset()
    print("________________TEST_GAME________________")
    for i in range(255):
        action, in_st = model.predict(obs.astype(np.float32), deterministic=False)
        # print(f"action: {action}")
        obs, reward, done, info = env.step(action)
        if i == 254 or done:
            env.render()
            break
            # _state = env.reset()
