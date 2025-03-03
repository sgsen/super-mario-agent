import gymnasium as gym

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    print(env.observation_space)
    print(env.action_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    print("CartPole-v1 environment loaded")


    # Test running one episode
    state, info = env.reset()
    print(state, info)
    print("Episode started")

    done = False
    
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(state, reward, done, truncated, info)
        done = done or truncated
    
    print("Episode finished")
    env.close()

if __name__ == "__main__":
    main() 