from stable_baselines3 import A2C

from env.checker import check_env
from env.gomoku_env import GomokuEnv
from policy.custom_cnn import custom_cnn_args

# Test environment
if __name__ == '__main__':
    env = GomokuEnv()
    reset = env.reset()
    check_env(env)

    model = A2C("CnnPolicy", env, policy_kwargs=custom_cnn_args, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(100):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
