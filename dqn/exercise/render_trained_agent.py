import torch
import gymnasium as gym

from dqn_agent import Agent


def main(nepisodes=10):
    # Initialize environment
    env = gym.make("LunarLander-v2", render_mode="human")

    # Initialize the RL agent
    agent = Agent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        seed=42,
    )

    # load the weights of the deep learning model from file
    agent.q_network_local.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(nepisodes):
        state, _ = env.reset()
        for j in range(400):
            action = agent.get_action(state)
            env.render()
            state, reward, done, _, _ = env.step(action)
            if done:
                break

    env.close()


if __name__ == "__main__":
    main()
