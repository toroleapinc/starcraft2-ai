"""Train SC2 agent."""
import argparse
import os
import torch
from env import SC2Env
from env.observation import preprocess_screen
from agents.a2c_agent import A2CAgent

def train(map_name='MoveToBeacon', episodes=3000):
    env = SC2Env(map_name=map_name)
    agent = A2CAgent()
    os.makedirs('checkpoints', exist_ok=True)
    best_reward = -float('inf')
    rewards_history = []
    for ep in range(episodes):
        obs = env.reset()
        screen = preprocess_screen(obs['screen'])
        episode_reward = 0
        done = False
        while not done:
            action_id, target = agent.select_action(screen)
            obs, reward, done = env.step(action_id, [[0], target])
            screen = preprocess_screen(obs['screen'])
            agent.rewards.append(reward)
            episode_reward += reward
        loss = agent.update()
        rewards_history.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy.state_dict(), 'checkpoints/best.pt')
        if (ep + 1) % 50 == 0:
            avg = sum(rewards_history[-50:]) / 50
            print(f"Ep {ep+1}: reward={episode_reward:.1f}, avg50={avg:.1f}, best={best_reward:.1f}")
    env.close()
# TODO: try PPO instead of A2C

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default='MoveToBeacon')
    parser.add_argument('--episodes', type=int, default=3000)
    args = parser.parse_args()
    train(args.map, args.episodes)
