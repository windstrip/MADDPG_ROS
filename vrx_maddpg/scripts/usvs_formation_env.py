#!/usr/bin/env python

import torch
import numpy as np
from maddpg import Agent
import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
import threading
import time

x1, y1, vx1, vy1 = 0., 0., 0., 0.
x2, y2, vx2, vy2 = 0., 0., 0., 0.
x3, y3, vx3, vy3 = 0., 0., 0., 0.
r1, p1, y1 = 0., 0., 0.
r2, p2, y2 = 0., 0., 0.
r3, p3, y3 = 0., 0., 0.

def processPositionDataWamv1(data):
    global x1, y1, vx1, vy1
    x1 = data.pose.pose.position.x
    y1 = data.pose.pose.position.y
    vx1 = data.twist.twist.linear.x
    vy1 = data.twist.twist.linear.y
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w
    r1 = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    p1 = np.arcsin(2*(w*y-z*x))
    y1 = np.arctan2(2*(w*z+x*y), 1-2*(z*z+y*y))
    
def processPositionDataWamv2(data):
    global x2, y2, vx2, vy2
    x2 = data.pose.pose.position.x
    y2 = data.pose.pose.position.y
    vx2 = data.twist.twist.linear.x
    vy2 = data.twist.twist.linear.y
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w
    r2 = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    p2 = np.arcsin(2*(w*y-z*x))
    y2 = np.arctan2(2*(w*z+x*y), 1-2*(z*z+y*y))

def processPositionDataWamv3(data):
    global x3, y3, vx3, vy3
    x3 = data.pose.pose.position.x
    y3 = data.pose.pose.position.y
    vx3 = data.twist.twist.linear.x
    vy3 = data.twist.twist.linear.y
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w
    r3 = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    p3 = np.arcsin(2*(w*y-z*x))
    y3 = np.arctan2(2*(w*z+x*y), 1-2*(z*z+y*y))

class MultiShipEnv(object):
    terminal = False
    action_dim = 2
    state_dim = 8
    n_agents = 3
    action_bound = [-1, 1]
    goal_point = [0, 0]

    formation_distance = 50.
    formation_distance_error = 5.
    goal_distance_error = 10.
    collision_distance = 10.

    def __init__(self, goal_point=[0, 0]):
        self.goal_point = goal_point

    def step(self, action=0):
        s, r = self._get_state_and_reward()
        return s, r, self.terminal
    
    def _get_state_and_reward(self):
        global x1, y1, vx1, vy1
        global x2, y2, vx2, vy2
        global x3, y3, vx3, vy3
        global y1, y2, y3

        state = np.zeros(self.n_agents * self.state_dim)
        reward = np.zeros(self.n_agents)

        # state of usv1
        s1_1, s1_2 = vx1, vy1
        s1_3, s1_4 = self.goal_point[0] - x1, self.goal_point[1] - y1
        s1_5, s1_6 = x2 - x1, y2 - y1
        s1_7, s1_8 = x3 - x1, y3 - y1  

        # state of usv2
        s2_1, s2_2 = vx2, vy2
        s2_3, s2_4 = self.goal_point[0] - x2, self.goal_point[1] - y2
        s2_5, s2_6 = x1 - x2, y1 - y2
        s2_7, s2_8 = x3 - x2, y3 - y2
        
        # state of usv3
        s3_1, s3_2 = vx3, vy3
        s3_3, s3_4 = self.goal_point[0] - x3, self.goal_point[1] - y3
        s3_5, s3_6 = x1 - x3, y1 - y3
        s3_7, s3_8 = x2 - x3, y2 - y3

        state[0:self.state_dim] = np.array([s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8])
        state[self.state_dim:2*self.state_dim] = np.array([s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8])
        state[2*self.state_dim:3*self.state_dim] = np.array([s3_1, s3_2, s3_3, s3_4, s3_5, s3_6, s3_7, s3_8])

        # reward of distance to goal
        factor_goal = 1
        dg1 = np.sqrt(s1_3**2 + s1_4**2)
        dg2 = np.sqrt(s2_3**2 + s2_4**2)
        dg3 = np.sqrt(s3_3**2 + s3_4**2)
        abs_v1 = np.sqrt(s1_1**2 + s1_2**2)
        abs_v2 = np.sqrt(s2_1**2 + s2_2**2)
        abs_v3 = np.sqrt(s3_1**2 + s3_2**2)
        rg1 = factor_goal*np.cos(np.arctan2(s1_4, s1_3) - y1)
        rg2 = factor_goal*np.cos(np.arctan2(s2_4, s2_3) - y2)
        rg3 = factor_goal*np.cos(np.arctan2(s3_4, s3_3) - y3)
        # rg1 = -factor_goal*dg1
        # rg2 = -factor_goal*dg2
        # rg3 = -factor_goal*dg3

        # reward of arriving goal point
        factor_success = 100
        ra1, ra2, ra3= 0., 0., 0.
        if dg1 <= self.goal_distance_error or dg2 <= self.goal_distance_error or dg3 <= self.goal_distance_error:
            self.terminal = True
            ra1 += factor_success
            ra2 += factor_success
            ra3 += factor_success

        # reward of formation
        factor_formation = 1.0
        d12 = np.sqrt(s1_5**2 + s1_6**2)
        d13 = np.sqrt(s1_7**2 + s1_8**2)
        d23 = np.sqrt(s2_7**2 + s2_8**2)
        rf1, rf2, rf3= 0., 0., 0.
        if abs(d12 - self.formation_distance) > self.formation_distance_error:
            rf1 += -factor_formation
            rf2 += -factor_formation
        if abs(d13 - self.formation_distance) > self.formation_distance_error:
            rf1 += -factor_formation
            rf3 += -factor_formation
        if abs(d23 - self.formation_distance) > self.formation_distance_error:
            rf2 += -factor_formation
            rf3 += -factor_formation
        
        # reward of collision
        rc1, rc2, rc3= 0., 0., 0.
        factor_collision = 10.0
        if d12 < self.collision_distance:
            rc1 += -factor_collision
            rc2 += -factor_collision
        if d13 < self.collision_distance:
            rc1 += -factor_collision
            rc3 += -factor_collision
        if d23 < self.collision_distance:
            rc2 += -factor_collision
            rc3 += -factor_collision

        reward[0] = rg1 + ra1 #+ rf1 + rc1
        reward[1] = rg2 + ra2 #+ rf2 + rc2
        reward[2] = rg3 + ra3 #+ rf3 + rc3

        return state, reward

    def reset(self):
        self.terminal = False
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
    goal_point = [0, 200]
    env = MultiShipEnv(goal_point)
    def run(self):
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

        rate = rospy.Rate(1) # 10hz

        left_cmd1, right_cmd1, left_angle1, right_angle1 = 0., 0., 0., 0.
        left_cmd2, right_cmd2, left_angle2, right_angle2 = 0., 0., 0., 0.
        left_cmd3, right_cmd3, left_angle3, right_angle3 = 0., 0., 0., 0.

        ON_TRAIN = True
        MAX_EPISODES = 10000
        MAX_EP_STEPS = 200

        n_agents = self.env.n_agents
        agent = Agent(env=self.env, n_agents=n_agents, random_seed=2)

        while not self.stopped:
            print('start learn')
            for i_episode in range(1, MAX_EPISODES + 1):
                s = self.env.reset()
                agent.reset()
                reset_world()
                scores = np.zeros(n_agents)
                for t in range(1, MAX_EP_STEPS + 1):
                    # step gazebo
                    a = agent.act(s)                                      
                    
                    # left_cmd1, right_cmd1, left_angle1, right_angle1 = a[0], a[0], a[1]*np.pi/2, a[1]*np.pi/2
                    # left_cmd2, right_cmd2, left_angle2, right_angle2 = a[2], a[2], a[3]*np.pi/2, a[3]*np.pi/2
                    # left_cmd3, right_cmd3, left_angle3, right_angle3 = a[4], a[4], a[5]*np.pi/2, a[5]*np.pi/2

                    left_cmd1, right_cmd1, left_angle1, right_angle1 = 1, 1, a[1]*0.1, a[1]*0.1
                    left_cmd2, right_cmd2, left_angle2, right_angle2 = 1, 1, a[3]*0.1, a[3]*0.1
                    left_cmd3, right_cmd3, left_angle3, right_angle3 = 1, 1, a[5]*0.1, a[5]*0.1

                    pub_left_cmd1.publish(left_cmd1)
                    pub_right_cmd1.publish(right_cmd1)
                    pub_left_ang1.publish(left_angle1)
                    pub_right_ang1.publish(right_angle1)

                    pub_left_cmd2.publish(left_cmd2)
                    pub_right_cmd2.publish(right_cmd2)
                    pub_left_ang2.publish(left_angle2)
                    pub_right_ang2.publish(right_angle2)

                    pub_left_cmd3.publish(left_cmd3)
                    pub_right_cmd3.publish(right_cmd3)
                    pub_left_ang3.publish(left_angle3)
                    pub_right_ang3.publish(right_angle3)

                    rate.sleep()  

                    # step agent
                    s_, r, done = self.env.step(a)
                    scores += r
                    agent.step(s, a, r, s_, done)
                    s = s_

                    # print
                    if done or t == MAX_EP_STEPS:
                        print('Ep:', i_episode,
                            '| Steps: %i' % t,
                            '| Reward: ', scores,
                            '| Terminal: %i' % self.env.terminal,
                            )
                        break

                # save network parameters
                if i_episode % 100 == 0:
                    for m in range(n_agents):
                        name = 'agent_' + str(m) + '_'
                        torch.save(agent.actor_local[m].state_dict(), name + 'checkpoint_actor.pth')
                        torch.save(agent.critic_local[m].state_dict(), name + 'checkpoint_critic.pth')    

            print('end learn')
            self.stopped = True
    def stop(self):
        self.stopped = True
    def isStopped(self):
        return self.stopped



if __name__ == '__main__':
    rospy.init_node('GetDataFromGazebo', anonymous=True)
    rospy.Subscriber('/wamv1/sensors/position/p3d_wamv', Odometry, processPositionDataWamv1)
    rospy.Subscriber('/wamv2/sensors/position/p3d_wamv', Odometry, processPositionDataWamv2)
    rospy.Subscriber('/wamv3/sensors/position/p3d_wamv', Odometry, processPositionDataWamv3)
    
    show_thread = TrainingThread()
    show_thread.start()

    rospy.spin()

    show_thread.stop()
    show_thread.join()
    