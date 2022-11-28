import numpy as np
from stable_baselines3 import A2C, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.dqn import MlpPolicy

from env.checker import check_env
from env.gomoku_env import GomokuEnv
from policy.custom_cnn import custom_cnn_args
from wandb.integration.sb3 import WandbCallback
# import wandb

# Train model and test it
if __name__ == '__main__':
    # wandb.init()
    env = GomokuEnv()
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=float(1.0) * np.ones(2))

    model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise)
    model.learn(total_timesteps=400000)
    model.save("ddpg_mountain")

    obs = env.reset()
    _state = obs
    for i in range(255):
        action, _state = model.predict(_state, deterministic=True)
        _state, reward, done, info = env.step(action)
        if i == 254 or done:
            env.render()
            _state = env.reset()

