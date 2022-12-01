import numpy as np
from stable_baselines3 import SAC, A2C
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
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=float(0.05) * np.ones(2))

    # model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise)
    model = A2C("MlpPolicy", env, verbose=1)
    # model = DDPG.load("ddpg_mountain", action_noise=action_noise)

    model.learn(total_timesteps=10000)
    model.save("discrete-life")

    obs = env.reset()
    print("________________TEST_GAME________________")
    for i in range(255):
        action, in_st = model.predict(obs.astype(np.float32), deterministic=False)
        obs, reward, done, info = env.step(action)
        if i == 254 or done:
            env.render()
            break
            # _state = env.reset()
