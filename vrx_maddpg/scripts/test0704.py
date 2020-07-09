#!/usr/bin/env python

from MADDPG import MADDPG
import numpy as np
import torch as th
import visdom
import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
import threading
import time
import copy
from tensorboardX import SummaryWriter
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

usv_info1 = np.zeros(8)
usv_info2 = np.zeros(8)
usv_info3 = np.zeros(8)
# x1, y1, vx1, vy1 = 0., 0., 0., 0.
# x2, y2, vx2, vy2 = 0., 0., 0., 0.
# x3, y3, vx3, vy3 = 0., 0., 0., 0.
# r1, p1, y1 = 0., 0., 0.
# r2, p2, y2 = 0., 0., 0.
# r3, p3, y3 = 0., 0., 0.

def processPositionDataWamv1(data):
    global usv_info1
    usv_info1[0] = data.pose.pose.position.x
    usv_info1[1] = data.pose.pose.position.y
    usv_info1[2] = data.twist.twist.linear.x
    usv_info1[3] = data.twist.twist.linear.y
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w
    usv_info1[4] = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    usv_info1[5] = np.arcsin(2*(w*y-z*x))
    usv_info1[6] = np.arctan2(2*(w*z+x*y), 1-2*(z*z+y*y))
    usv_info1[7] = data.twist.twist.angular.z
    
def processPositionDataWamv2(data):
    global usv_info2
    usv_info2[0] = data.pose.pose.position.x
    usv_info2[1] = data.pose.pose.position.y
    usv_info2[2] = data.twist.twist.linear.x
    usv_info2[3] = data.twist.twist.linear.y
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w
    usv_info2[4] = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    usv_info2[5] = np.arcsin(2*(w*y-z*x))
    usv_info2[6] = np.arctan2(2*(w*z+x*y), 1-2*(z*z+y*y))
    usv_info2[7] = data.twist.twist.angular.z

def processPositionDataWamv3(data):
    global usv_info3
    usv_info3[0] = data.pose.pose.position.x
    usv_info3[1] = data.pose.pose.position.y
    usv_info3[2] = data.twist.twist.linear.x
    usv_info3[3] = data.twist.twist.linear.y
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w
    usv_info3[4] = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    usv_info3[5] = np.arcsin(2*(w*y-z*x))
    usv_info3[6] = np.arctan2(2*(w*z+x*y), 1-2*(z*z+y*y))
    usv_info3[7] = data.twist.twist.angular.z

class MultiShipEnv(object):
    terminal = False
    action_dim = 2
    state_dim = 10
    n_agents = 3
    action_bound = [-1, 1]
    goal_point = [0, 0]

    formation_distance = 50.
    formation_distance_error = 5.
    goal_distance_error = 10.
    collision_distance = 10.
    action = np.zeros((n_agents, action_dim))

    def __init__(self, goal_point=[0, 0]):
        self.goal_point = goal_point

    def step(self, action=0):
        self.action = copy.copy(action)
        s, r = self._get_state_and_reward()
        return s, r, self.terminal, []
    
    def _get_state_and_reward(self):
        global usv_info1
        global usv_info2
        global usv_info3
        
        x1, y1, vx1, vy1, r1, p1, h1, az1 = usv_info1[0], usv_info1[1], usv_info1[2], usv_info1[3], usv_info1[4], usv_info1[5], usv_info1[6], usv_info1[7]
        x2, y2, vx2, vy2, r2, p2, h2, az2 = usv_info2[0], usv_info2[1], usv_info2[2], usv_info2[3], usv_info2[4], usv_info2[5], usv_info2[6], usv_info2[7]
        x3, y3, vx3, vy3, r3, p3, h3, az3 = usv_info3[0], usv_info3[1], usv_info3[2], usv_info3[3], usv_info3[4], usv_info3[5], usv_info3[6], usv_info3[7]

        state = []
        reward = np.zeros((self.n_agents,))

        # state of usv1
        s1_1, s1_2 = self.goal_point[0] - x1, self.goal_point[1] - y1
        s1_3, s1_4 = x2 - x1, y2 - y1
        s1_5, s1_6 = x3 - x1, y3 - y1
        s1_7, s1_8 = vx1, vy1
        s1_9, s1_10 = h1, az1

        # state of usv2
        s2_1, s2_2 = self.goal_point[0] - x2, self.goal_point[1] - y2
        s2_3, s2_4 = x1 - x2, y1 - y2
        s2_5, s2_6 = x3 - x2, y3 - y2
        s2_7, s2_8 = vx2, vy2
        s2_9, s2_10 = h2, az2

        # state of usv3
        s3_1, s3_2 = self.goal_point[0] - x3, self.goal_point[1] - y3
        s3_3, s3_4 = x1 - x3, y1 - y3
        s3_5, s3_6 = x2 - x3, y2 - y3
        s3_7, s3_8 = vx3, vy3
        s3_9, s3_10 = h3, az3

        state.append([s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8, s1_9, s1_10])
        state.append([s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9, s2_10])
        state.append([s3_1, s3_2, s3_3, s3_4, s3_5, s3_6, s3_7, s3_8, s3_9, s3_10])

        # reward of distance to goal
        factor_goal = 0.05
        rg1, rg2, rg3 = 0., 0., 0.
        dg1 = np.sqrt(s1_1**2 + s1_2**2)
        dg2 = np.sqrt(s2_1**2 + s2_2**2)
        dg3 = np.sqrt(s3_1**2 + s3_2**2)
        rg1 = -factor_goal*dg1
        rg2 = -factor_goal*dg2
        rg3 = -factor_goal*dg3

        # reward of arriving goal point
        factor_success = 100.0
        ra1, ra2, ra3 = 0., 0., 0.
        if dg1 <= self.goal_distance_error or dg2 <= self.goal_distance_error or dg3 <= self.goal_distance_error:
            self.terminal = True
            ra1 += factor_success
            ra2 += factor_success
            ra3 += factor_success

        # reward of formation
        factor_formation = 0.1
        d12 = np.sqrt(s1_3**2 + s1_4**2)
        d13 = np.sqrt(s1_5**2 + s1_6**2)
        d23 = np.sqrt(s2_5**2 + s2_6**2)
        rf1, rf2, rf3= 0., 0., 0.
        error_d12 = abs(d12 - self.formation_distance)
        if error_d12 > self.formation_distance_error:
            rf1 += -factor_formation*(error_d12 - self.formation_distance_error)
            rf2 += -factor_formation*(error_d12 - self.formation_distance_error)
        error_d13 = abs(d13 - self.formation_distance)
        if error_d13 > self.formation_distance_error:
            rf1 += -factor_formation*(error_d13 - self.formation_distance_error)
            rf3 += -factor_formation*(error_d13 - self.formation_distance_error)
        error_d23 = abs(d23 - self.formation_distance)
        if error_d23 > self.formation_distance_error:
            rf2 += -factor_formation*(error_d23 - self.formation_distance_error)
            rf3 += -factor_formation*(error_d23 - self.formation_distance_error)
        
        # reward of collision
        rc1, rc2, rc3 = 0., 0., 0.
        # factor_collision = 20.0
        # if d12 < self.collision_distance:
        #     rc1 += -factor_collision
        #     rc2 += -factor_collision
        # if d13 < self.collision_distance:
        #     rc1 += -factor_collision
        #     rc3 += -factor_collision
        # if d23 < self.collision_distance:
        #     rc2 += -factor_collision
        #     rc3 += -factor_collision
        
        # reward of action
        factor_action = 0.
        r_action1, r_action2, r_action3 = 0., 0., 0.
        # r_action1 = factor_action * (self.action[0, 0] - abs(self.action[0, 1]))
        # r_action2 = factor_action * (self.action[1, 0] - abs(self.action[1, 1]))
        # r_action3 = factor_action * (self.action[2, 0] - abs(self.action[2, 0]))

        reward[0] = rg1 + ra1 + rf1 + rc1 + r_action1
        reward[1] = rg2 + ra2 + rf2 + rc2 + r_action2
        reward[2] = rg3 + ra3 + rf3 + rc3 + r_action3
        return state, reward

    def reset(self):
        self.terminal = False
        self.action = np.zeros((self.n_agents, self.action_dim))
        s, _ = self._get_state_and_reward()
        return s

    def sample_action(self):
        all_action_size = self.n_agents * self.action_dim
        action = np.zeros(all_action_size)
        for i in range(all_action_size):
            action[i] = np.random.uniform(*self.action_bound)
        return action

    def get_action_dim(self, num):
        return self.action_dim

    def get_state_dim(self, num):
        return self.state_dim

class TrainingThread(threading.Thread):
    stopped = False
    ON_TRAIN = True
    MAX_EPISODES = 10000
    MAX_EP_STEPS = 200
    batch_size = 128
    capacity = 500000
    episodes_before_train = 100
    goal_point = [0, 200]
    env = MultiShipEnv(goal_point)

    # vis = visdom.Visdom(port=5274)

    n_agents = env.n_agents
    n_states = env.state_dim
    n_actions = env.action_dim
    maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train)

    pub_left_cmd1 = rospy.Publisher('/wamv1/thrusters/left_thrust_cmd', Float32, queue_size=10)
    pub_right_cmd1 = rospy.Publisher('/wamv1/thrusters/right_thrust_cmd', Float32, queue_size=10)
    pub_left_ang1 = rospy.Publisher('/wamv1/thrusters/left_thrust_angle', Float32, queue_size=10)
    pub_right_ang1 = rospy.Publisher('/wamv1/thrusters/right_thrust_angle', Float32, queue_size=10)

    pub_left_cmd2 = rospy.Publisher('/wamv2/thrusters/left_thrust_cmd', Float32, queue_size=10)
    pub_right_cmd2 = rospy.Publisher('/wamv2/thrusters/right_thrust_cmd', Float32, queue_size=10)
    pub_left_ang2 = rospy.Publisher('/wamv2/thrusters/left_thrust_angle', Float32, queue_size=10)
    pub_right_ang2 = rospy.Publisher('/wamv2/thrusters/right_thrust_angle', Float32, queue_size=10)

    pub_left_cmd3 = rospy.Publisher('/wamv3/thrusters/left_thrust_cmd', Float32, queue_size=10)
    pub_right_cmd3 = rospy.Publisher('/wamv3/thrusters/right_thrust_cmd', Float32, queue_size=10)
    pub_left_ang3 = rospy.Publisher('/wamv3/thrusters/left_thrust_angle', Float32, queue_size=10)
    pub_right_ang3 = rospy.Publisher('/wamv3/thrusters/right_thrust_angle', Float32, queue_size=10)

    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    set_physics_properties = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
    get_physics_properties = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)

    physics_properties = get_physics_properties()
    set_physics_properties(0.01, 10000, physics_properties.gravity, physics_properties.ode_config)

    rate = []

    left_cmd1, right_cmd1, left_angle1, right_angle1 = 0., 0., 0., 0.
    left_cmd2, right_cmd2, left_angle2, right_angle2 = 0., 0., 0., 0.
    left_cmd3, right_cmd3, left_angle3, right_angle3 = 0., 0., 0., 0.

    def run(self):
        self.rate = rospy.Rate(1) # 10hz
        while not self.stopped:
            if self.ON_TRAIN:
                self.train()
            else:
                self.value()
            self.stopped = True
    def train(self):
        print('start learn')
        FloatTensor = th.cuda.FloatTensor if self.maddpg.use_cuda else th.FloatTensor
        reward_record = []
        win = None
        param = None

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        log_file_name = TIMESTAMP
        writer = SummaryWriter(log_dir='train')

        for i_episode in range(self.MAX_EPISODES):
            self.reset_world()
            time.sleep(0.1) # 保证reset_world()执行之后，传感器数据可以更新为最新
            obs = self.env.reset()
            obs = np.stack(obs)
            if isinstance(obs, np.ndarray):
                obs = th.from_numpy(obs).float()
            total_reward = 0.0
            rr = np.zeros((self.n_agents,))
            for t in range(self.MAX_EP_STEPS):
                obs = obs.type(FloatTensor)
                action = self.maddpg.select_action(obs).data.cpu()

                a = action.numpy()
                # left_cmd1, right_cmd1, left_angle1, right_angle1 = a[0, 0], a[0, 0], a[0, 1]*0.1, a[0, 1]*0.1
                # left_cmd2, right_cmd2, left_angle2, right_angle2 = a[1, 0], a[1, 0], a[1, 1]*0.1, a[1, 1]*0.1
                # left_cmd3, right_cmd3, left_angle3, right_angle3 = a[2, 0], a[2, 0], a[2, 1]*0.1, a[2, 1]*0.1
                left_cmd1, right_cmd1, left_angle1, right_angle1 = (a[0, 0]+1.0)/2.0, (a[0, 0]+1.0)/2.0, a[0, 1]*0.1, a[0, 1]*0.1
                left_cmd2, right_cmd2, left_angle2, right_angle2 = (a[1, 0]+1.0)/2.0, (a[1, 0]+1.0)/2.0, a[1, 1]*0.1, a[1, 1]*0.1
                left_cmd3, right_cmd3, left_angle3, right_angle3 = (a[2, 0]+1.0)/2.0, (a[2, 0]+1.0)/2.0, a[2, 1]*0.1, a[2, 1]*0.1
                # left_cmd1, right_cmd1, left_angle1, right_angle1 = 1.0, 1.0, a[0, 1]*0.1, a[0, 1]*0.1
                # left_cmd2, right_cmd2, left_angle2, right_angle2 = 1.0, 1.0, a[1, 1]*0.1, a[1, 1]*0.1
                # left_cmd3, right_cmd3, left_angle3, right_angle3 = 1.0, 1.0, a[2, 1]*0.1, a[2, 1]*0.1

                self.pub_left_cmd1.publish(left_cmd1)
                self.pub_right_cmd1.publish(right_cmd1)
                self.pub_left_ang1.publish(left_angle1)
                self.pub_right_ang1.publish(right_angle1)

                self.pub_left_cmd2.publish(left_cmd2)
                self.pub_right_cmd2.publish(right_cmd2)
                self.pub_left_ang2.publish(left_angle2)
                self.pub_right_ang2.publish(right_angle2)

                self.pub_left_cmd3.publish(left_cmd3)
                self.pub_right_cmd3.publish(right_cmd3)
                self.pub_left_ang3.publish(left_angle3)
                self.pub_right_ang3.publish(right_angle3)

                self.rate.sleep() 

                obs_, reward, done, _ = self.env.step(action.numpy())

                reward = th.FloatTensor(reward).type(FloatTensor)
                obs_ = np.stack(obs_)
                obs_ = th.from_numpy(obs_).float()
                if t != self.MAX_EP_STEPS - 1:
                    next_obs = obs_
                else:
                    next_obs = None

                total_reward += reward.sum()
                rr += reward.cpu().numpy()
                self.maddpg.memory.push(obs.data, action, next_obs, reward)
                obs = next_obs

                c_loss, a_loss = self.maddpg.update_policy() 

                # print
                if done or t == self.MAX_EP_STEPS -1 :
                    print('Ep:', i_episode,
                        '| Steps: %i' % t,
                        '| Reward: ', rr,
                        '| Terminal: %i' % done,
                        '| Var:', self.maddpg.var
                        )
                    break
            self.maddpg.episode_done += 1
            # print('Episode: %d, reward = %f' % (i_episode, total_reward))
            reward_record.append(total_reward)
            writer.add_scalars(log_file_name, {'Steps': t + 1, 'r1': rr[0], 'r2': rr[1], 'r3': rr[2], 
                                                }, global_step=i_episode)

            if (i_episode + 1) % 500 == 0:
                for m in range(self.n_agents):
                    name = str(i_episode + 1) + '_' + 'agent_' + str(m) + '_'
                    th.save(self.maddpg.actors[m].state_dict(), name + 'checkpoint_actor.pth')
                    th.save(self.maddpg.critics[m].state_dict(), name + 'checkpoint_critic.pth')

            # if win is None:
            #     win = self.vis.line(X=np.arange(i_episode, i_episode+1),
            #                         Y=np.array([np.append(total_reward, rr)]),
            #                         opts=dict(
            #                             ylabel='Reward',
            #                             xlabel='Episode',
            #                             title='MADDPG on USV Formation',
            #                             legend=['Total'] +
            #                             ['Agent-%d' % i for i in range(self.n_agents)]))
            # else:
            #     self.vis.line(X=np.array([np.array(i_episode).repeat(self.n_agents+1)]),
            #                   Y=np.array([np.append(total_reward, rr)]),
            #                   win=win,
            #                   update='append')
            # if param is None:
            #     param = self.vis.line(X=np.arange(i_episode, i_episode+1),
            #                           Y=np.array([self.maddpg.var[0]]),
            #                           opts=dict(
            #                               ylabel='Var',
            #                               xlabel='Episode',
            #                               title='MADDPG on USV Formation: Exploration',
            #                               legend=['Variance']))
            # else:
            #     self.vis.line(X=np.array([i_episode]),
            #                   Y=np.array([self.maddpg.var[0]]),
            #                   win=param,
            #                   update='append')          

        print('end learn')
        writer.close()
    def value(self):
        test_episode = 1000
        network_time = '2020-07-06T22-18-03'
        network_path = './train/' + network_time + '/network_parameters/'
        for m in range(self.n_agents):
            name = network_path + str(test_episode) + '_' + 'agent_' + str(m) + '_'
            self.maddpg.actors[m].load_state_dict(th.load(name + 'checkpoint_actor.pth'))
        print('start test')
        FloatTensor = th.cuda.FloatTensor if self.maddpg.use_cuda else th.FloatTensor
        for i_episode in range(20):
            self.reset_world()
            time.sleep(0.1) # 保证reset_world()执行之后，传感器数据可以更新为最新
            obs = self.env.reset()
            obs = np.stack(obs)
            if isinstance(obs, np.ndarray):
                obs = th.from_numpy(obs).float()
            rr = np.zeros((self.n_agents,))
            for t in range(self.MAX_EP_STEPS):
                obs = obs.type(FloatTensor)
                action = self.maddpg.select_action(obs, noise=False).data.cpu()

                a = action.numpy()
                left_cmd1, right_cmd1, left_angle1, right_angle1 = (a[0, 0]+1.0)/2.0, (a[0, 0]+1.0)/2.0, a[0, 1]*0.1, a[0, 1]*0.1
                left_cmd2, right_cmd2, left_angle2, right_angle2 = (a[1, 0]+1.0)/2.0, (a[1, 0]+1.0)/2.0, a[1, 1]*0.1, a[1, 1]*0.1
                left_cmd3, right_cmd3, left_angle3, right_angle3 = (a[2, 0]+1.0)/2.0, (a[2, 0]+1.0)/2.0, a[2, 1]*0.1, a[2, 1]*0.1

                self.pub_left_cmd1.publish(left_cmd1)
                self.pub_right_cmd1.publish(right_cmd1)
                self.pub_left_ang1.publish(left_angle1)
                self.pub_right_ang1.publish(right_angle1)

                self.pub_left_cmd2.publish(left_cmd2)
                self.pub_right_cmd2.publish(right_cmd2)
                self.pub_left_ang2.publish(left_angle2)
                self.pub_right_ang2.publish(right_angle2)

                self.pub_left_cmd3.publish(left_cmd3)
                self.pub_right_cmd3.publish(right_cmd3)
                self.pub_left_ang3.publish(left_angle3)
                self.pub_right_ang3.publish(right_angle3)

                self.rate.sleep() 

                obs, reward, done, _ = self.env.step(action.numpy())

                reward = th.FloatTensor(reward).type(FloatTensor)
                obs = np.stack(obs)
                obs = th.from_numpy(obs).float()
                rr += reward.cpu().numpy()
                # print
                if done or t == self.MAX_EP_STEPS -1 :
                    print('Ep:', i_episode,
                        '| Steps: %i' % t,
                        '| Reward: ', rr,
                        '| Terminal: %i' % done,
                        )
                    break     
        print('end test')
    def stop(self):
        self.stopped = True
    def isStopped(self):
        return self.stopped


if __name__ == '__main__':
    rospy.init_node('GetDataFromGazebo', anonymous=True)
    rospy.Subscriber('/wamv1/sensors/position/p3d_wamv', Odometry, processPositionDataWamv1)
    rospy.Subscriber('/wamv2/sensors/position/p3d_wamv', Odometry, processPositionDataWamv2)
    rospy.Subscriber('/wamv3/sensors/position/p3d_wamv', Odometry, processPositionDataWamv3)
    
    train_thread = TrainingThread()
    train_thread.start()

    rospy.spin()

    train_thread.stop()
    train_thread.join()

