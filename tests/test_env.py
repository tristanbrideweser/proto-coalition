from hemac import HeMAC_v0

env = HeMAC_v0.env(render_mode="human")
env.reset(seed=0)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
    env.step(action)
env.close()