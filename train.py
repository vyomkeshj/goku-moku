from stable_baselines3 import A2C

from env.checker import check_env
from env.gomoku_env import GomokuEnv
from policy.custom_cnn import custom_cnn_args
from wandb.integration.sb3 import WandbCallback
import wandb
# Train model and test it
if __name__ == '__main__':
    wandb.init()
    env = GomokuEnv()
    check_env(env)

    model = A2C("CnnPolicy", env, policy_kwargs=custom_cnn_args, verbose=1)
    model.learn(total_timesteps=10000, callback=WandbCallback())

    obs = env.reset()
    print(obs.shape)
    for i in range(255):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
