# PZ2

## Motivation
Consider the following scenarios:
1. A single-agent environment
2. A multi-agent environment with simultaneous actions
3. A multi-agent environment with alternating actions
4. A multi-agent environment with a custom sequence of actions (the sequence, considered part of the environment, could be stationary or changing)

Some of these scenarios have pre-existing APIs:
1. Gymnasium.env
2. PettingZoo.parallel_env
3. PettingZoo.AECEnv
4. No current API

However, 1 is a subset of 2, 2 is a subset of 3, and 3 is a subset of 4. We have been aware of this for years but, given that Gym and PettingZoo were managed by different orgs, it was impossible to unify.

This proposal provides an API for the most general scenario (4), whilst keeping the familiarity and simplicity of the Gymnasium API.

## Example use
```python
import pz2 as pz
env = pz.make("TicTacToe-v0")
observation, info = env.reset(seed=42)

policies = {env.agents[0]: circle_policy,
            env.agents[1]: cross_policy}
actives = {env.agents[0]: True,
           env.agents[1]: False}

while env.agents:
    actions = {agent: None for agent in env.agents}
    for agent in env.agents:
        if actives[agent]:
            actions[agent] = policies[agent](observation[agent])
    observation, rewards, terminations, truncations, next_active_agents, info = env.step(actions)
env.close()
```

### Comparison with Gymnasium
Using Gymnasium's API looks like
```python
import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
while not truncated or terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
env.close()
```
The only difference is that the dimension of the actions, observations, rewards, terminated, truncated and info are multi-dimensional.

## Advanced: Dealing with non-simultaneous actions
In the case of non-simultaneous actions, multiple `env.step()` calls will be made. This leads to the question of how to handle rewards when a non-simultaneous environment terminates. For example, when tic-tac-toe terminates on Player 2's move (Player 2 wins), Player 1 still needs to be allocated a reward for a loss. (WIP)
