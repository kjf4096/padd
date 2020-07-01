import paddle.fluid as fluid
import parl
from parl import layers

import copy
import numpy as np
import os
import gym
from parl.utils import logger
import turtle as t

class Paddle():

    def __init__(self):

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0

        # Setup Background

        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # Paddle

        self.paddle = t.Turtle()
        self.paddle.speed(10)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # Ball

        self.ball = t.Turtle()
        self.ball.speed(10)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 3
        self.ball.dy = -3

        # Score

        self.score = t.Turtle()
        self.score.speed(10)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------

        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle movement

    def paddle_right(self):

        x = self.paddle.xcor()
        if x < 225:
            self.paddle.setx(x+20)

    def paddle_left(self):

        x = self.paddle.xcor()
        if x > -225:
            self.paddle.setx(x-20)

    # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):

        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)
        return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

    def step(self, action):

        self.reward = 0
        self.done = 0

        if action == 0:
            self.paddle_left()
            self.reward -= .001

        if action == 2:
            self.paddle_right()
            self.reward -= .001

        self.run_frame()

        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        return self.reward, state, self.done

    def run_frame(self):

        self.win.update()

        # Ball moving

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision

        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward -= 1
            self.done = True

        # Ball Paddle collision

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward += 3


BATCH_SIZE = 128   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0001 # 学习率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
class Model(parl.Model):
    def __init__(self, act_dim):
        ######################################################################
        ######################################################################
        #
        hid1_size = 128
        hid2_size = 128
        
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        
        self.fc3 = layers.fc(size=act_dim, act=None)
        # 2. 请参考课堂Demo，配置model
        #
        ######################################################################
        ######################################################################

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        
        ######################################################################
        ######################################################################
        #
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        
        Q = self.fc3(h2)
       
        # 3. 请参考课堂Demo，组装Q网络
        #
        ######################################################################
        ######################################################################
        return Q

from parl.algorithms import DQN # 直接从parl库中导入DQN算法，无需自己重写算法
class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost
		
# replay_memory.py
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0

    
    while True:
    
        step += 1
        
        
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        reward,next_obs, done = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))
        # train model
        
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done
        
        total_reward += reward
        obs = next_obs
       
        
        if done:
            
            break


    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def run_episode2(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0

        
        
    
    
    while True:
    
        step += 1
        
        
        action = agent.predict(obs)  # 采样动作，所有动作都有概率被尝试到
        reward,next_obs, done = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))
        # train model
        '''
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done
        '''
        total_reward += reward
        obs = next_obs
       
        
        if done:
            #print(total_reward)
            break
        #if total_reward > 63:
            
        #    break

    return total_reward


def evaluate(env, agent):
    eval_reward = []
    
    for i in range(5):
        e_reward=0
        e_reward=run_episode2(env, agent, rpm)
                        
        eval_reward.append(e_reward)
        
    return np.mean(eval_reward)


env = Paddle()
np.random.seed(0)
action_dim = 3
obs_shape = 5
# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池



# 根据parl框架构建agent
######################################################################
######################################################################
#
# 4. 请参考课堂Demo，嵌套Model, DQN, Agent构建 agent
#
######################################################################
######################################################################
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
        algorithm,
        obs_dim=5,
        act_dim=action_dim,
        e_greed=0.1,  # 有一定概率随机选取动作，探索
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低



# 加载模型
if os.path.exists('./model.ckpt'):
    agent.restore('./model.ckpt')
    #print('loaded')


# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 20000

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    eval_reward = evaluate(env, agent)  
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))
    for i in range(0, 20):
        total_reward = run_episode(env, agent, rpm)
        episode += 1
        



    # test part
    eval_reward = evaluate(env, agent)  
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))
		

    # 训练结束，保存模型
    save_path ='./model.ckpt'.format(eval_reward)
    
    agent.save(save_path)



