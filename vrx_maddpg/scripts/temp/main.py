import torch
import numpy as np
from usvs_formation_env_without_obstacle import MultiShipEnv
from maddpg import Agent
from tensorboardX import SummaryWriter
from datetime import datetime

NO_REUSE = 0    # no reuse
REUSE_1 = 1     # 3 ships
REUSE_2 = 2     # 5 ships
REUSE = NO_REUSE

ON_TRAIN = True
RENDER = False
RE_TRAIN = False
MAX_EPISODES = 100000
MAX_EP_STEPS = 200
VAR_MIN = 0.1

env = MultiShipEnv(discrete_action=False)
n_agents = env.n_agents
agent = Agent(env=env, n_agents=n_agents, random_seed=2)

def train():
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    log_file_name = TIMESTAMP
    writer = SummaryWriter(log_dir='train')

    if RE_TRAIN:
        for m in range(n_agents):
            name = 'agent_' + str(m) + '_'
            agent.actor_local[m].load_state_dict(torch.load(name + 'checkpoint_actor.pth'))
            agent.critic_local[m].load_state_dict(torch.load(name + 'checkpoint_critic.pth'))

    for i_episode in range(1, MAX_EPISODES + 1):
        s = env.reset()
        agent.reset()
        scores = np.zeros(n_agents)
        for t in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
            a = agent.act(s)
            s_, r, done = env.step(a)
            scores += r

            agent.step(s, a, r, s_, done)
            s = s_

            if done or t == MAX_EP_STEPS-1:
                print('Ep:', i_episode,
                      '| Steps: %i' % int(t + 1),
                      '| Reward: ', scores,
                      '| Terminal: %i' % env.terminal,
                      )
                break
        if i_episode % 100 == 0:
            for m in range(n_agents):
                name = 'agent_' + str(m) + '_'
                torch.save(agent.actor_local[m].state_dict(), name + 'checkpoint_actor.pth')
                torch.save(agent.critic_local[m].state_dict(), name + 'checkpoint_critic.pth')
        writer.add_scalars(log_file_name + '/train', {'Steps': t + 1,
                                                      'L_d1': env.reward_per_episode[0][0], 'L_phi1': env.reward_per_episode[0][1], 'L_r1': env.reward_per_episode[0][-1],
                                                      'F_d1': env.reward_per_episode[1][0], 'F_phi1': env.reward_per_episode[1][1], 'F_r1': env.reward_per_episode[1][-1],
                                                      'F_d2': env.reward_per_episode[2][0], 'F_phi2': env.reward_per_episode[2][1], 'F_r2': env.reward_per_episode[2][-1],
                                                      'F_d3': env.reward_per_episode[3][0], 'F_phi3': env.reward_per_episode[3][1], 'F_r3': env.reward_per_episode[3][-1],
                                                      'F_d4': env.reward_per_episode[4][0], 'F_phi4': env.reward_per_episode[4][1], 'F_r4': env.reward_per_episode[4][-1],
                                                      }, global_step=i_episode)
    writer.close()


def eval():
    env.set_fps(10)
    for m in range(n_agents):
        ship_type = env.ship_info[m][-2]
        if REUSE == NO_REUSE:
            name = 'agent_' + str(m) + '_'
        elif REUSE == REUSE_1:
            if m <= 0:
                name = 'agent_' + str(m) + '_'
            else:
                if ship_type == 1:
                    name = 'agent_1_'
                else:
                    name = 'agent_2_'
        elif REUSE == REUSE_2:
            if m <= 0:
                name = 'agent_' + str(m) + '_'
            else:
                if ship_type == 1:
                    name = 'agent_3_'
                else:
                    name = 'agent_4_'

        agent.actor_local[m].load_state_dict(torch.load(name + 'checkpoint_actor.pth'))

    while True:
        s = env.reset()
        for t in range(200):
            env.render()
            a = agent.act(s, add_noise=False)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()