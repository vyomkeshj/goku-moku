import numpy as np
from stable_baselines3 import A2C

from env.checker import check_env
from env.gomoku_env import GomokuEnv
from policy.custom_cnn import custom_cnn_args
from wandb.integration.sb3 import WandbCallback
# import wandb

# Train model and test it
if __name__ == '__main__':
    # wandb.init()
    env = GomokuEnv()
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    _state = obs
    for i in range(255):
        action, _state = model.predict(_state, deterministic=True)
        _state, reward, done, info = env.step(action)
        if i == 254 or done:
            env.render()
            _state = env.reset()

