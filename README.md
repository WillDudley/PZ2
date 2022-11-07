# PZ2

## Motivation
Consider the following scenarios:
1. A single-agent environment
2. A multi-agent environment with simultaneous actions
3. A multi-agent environment with alternating actions
4. A multi-agent environment with a custom sequence of actions (the sequence could be stationary or changing)

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
env = pz.make("KnightArcherZombies-v10")
policies = {agent_name: policy, ...}

observation, info = env.reset(seed=42)
while env.active:
    actions = {agent_name: policies[agent_name](agent_obs) for agent_name, agent_obs in obs.items()}
    obs, rewards, terminations, truncations, info = env.step(actions)
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

## Dealing with non-simultaneous actions
In the case of non-simultaneous actions, multiple `env.step()` calls will be made. 
