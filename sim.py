from env_utils import SawyerBin  
import env_utils as eu
import os
os.environ.pop('SDL_VIDEODRIVER', None) 


def simulate(env_name):
    env, obs_dim, max_steps = eu.load(env_name)
    
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    for step in range(max_steps):
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)
        print(f"Step {step+1} - Reward: {reward}, Done: {done}")

        if step % 10 == 0:
            env.render()

        if done:
            print("Episode ended early.")
            break
    
    input("Press Enter to close the environment and exit...")
    env.close()

if __name__ == "__main__":
    simulate('sawyer_bin')   
