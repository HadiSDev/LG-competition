from tensorforce import Agent, Environment

from game.simulator import Simulator

sim = Simulator(host="http://localhost:8090")
# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='lg-competition-v0', simulator=sim
)
k = environment.__getattr__("environment")
# Instantiate a Tensorforce agent

agent = Agent.create(agent='../configs/a2c.json', environment=environment)

# Train for 300 episodes
for _ in range(10):

    # Initialize episode
    states = environment.reset()
    terminal = False
    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

agent.close()
environment.close()