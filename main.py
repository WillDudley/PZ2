from pz2.classic.TicTacToe import TicTacToeEnv
env = TicTacToeEnv()
observation, info = env.reset(seed=42)

policies = {env.agents[0]: env.action_space(env.agents[0]).sample,
            env.agents[1]: env.action_space(env.agents[0]).sample}
actives = {env.agents[0]: True,
           env.agents[1]: False}

while env.agents:
    actions = {agent: None for agent in env.agents}
    for agent in env.agents:
        if actives[agent]:
            actions[agent] = policies[agent](mask=observation[agent]["action_mask"])
    observation, rewards, terminations, truncations, next_active_agents, info = env.step(actions)
env.close()
