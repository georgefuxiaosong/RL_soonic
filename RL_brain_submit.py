# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:59:49 2018

OpenAI提交版 RL_brain
OpenAI游戏RL_brain

@author: Fuxiao
"""
import gym
import numpy as np
import tensorflow as tf
import cv2
#import sys
from collections import deque
import random
#import math
#import retro

def batch_norm(xs):#归一化
        fc_mean, fc_var = tf.nn.moments(xs, axes=[0,1,2])  
        scale = tf.Variable(tf.ones([1]))  
        shift = tf.Variable(tf.zeros([1]))  
        epsilon = 0.001  
        
        ema = tf.train.ExponentialMovingAverage(decay=0.5)          #滑动窗口
        def mean_var_with_update():  
            ema_apply_op = ema.apply([fc_mean, fc_var])  
            with tf.control_dependencies([ema_apply_op]):  
                return tf.identity(fc_mean), tf.identity(fc_var)  
        mean, var = mean_var_with_update()  
        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)  
        return xs

class DeepQNetwork(gym.Wrapper):
    def __init__(
            self,
            n_actions,  #有多少个action
            env,
            dropout=0.75,
            #n_features, #有多少个state，接收多少个observation,比如有长、宽、高这三种，这就是3个observation
            learning_rate=1e-5, #算法的初始学习率
            reward_decay=0.95, #对下一个Q值的衰减率
            e_greedy=0.90, #贪心算法的利用率,百分之九十的概率利用
            replace_target_iter=800, #300步以后更换新、旧神经网络中的参数，隔了多少步以后Q现实target_net的参数变为最新的参数
            memory_size=30000, #所需记忆的步数，没有训练之前先进行观察，随机运动，然后把观察的结果弄下来作为训练数据，先观察500个回合然后保存
            explore=300000, #迭代的次数
            observe=10000, #当超过这个数目，就开始贪心算法的利用率，减少探索率
            #replay_memory=5000, #首先不是要进行观察吗？就让观察的结果存到REPLAY_MEMORY里面去
            batch_size=32, #batch大小
            #e_greedy_increment=None, #是否缩小贪心算法的探索率
            output_graph=False, #output_graph是否输出tensorboard文件
            double_q=True, #是否使用Double DQN
            sess=None):
        super(DeepQNetwork, self).__init__(env)
        
        self.n_actions = n_actions
        #self.env=env
        self.gama = reward_decay
        self.memory_size = memory_size
        self.dropout=dropout
        #self.n_features = n_features
        self.lr = learning_rate
        self.explore=explore
        self.epsilon = e_greedy
        self.replace_target_iter = replace_target_iter
        
        self.observe_pic=observe
        #self.replay=replay_memory
        self.batch_size = batch_size
        self.final_epsilon =0.001
        #self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter=0
        self.choose_act=0 #计算选择了多少次动作
        self.count_back=0 #后退了多少步
        self.turn_back=True #是否返回的标志
        self.last_memory=deque()
        self.count=0
        self.turn_right=0
        
        self.double_q = double_q    # 决定是否使用Double DQN
        
        np.random.seed(1)
        tf.set_random_seed(1)
        
        self.builtnet() #建立网络
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess 
        #tf.summary.FileWriter("G:/OpenAI/资料/Competition",sess.graph)
        self.memory=deque() #先对游戏进行观察，记录下观察值s， a, reward, s_, done 
        
    def weights_variable(shape): #权重初始化函数
        #shape:需要初始化的权重矩阵大小
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01)) #产生阶段高斯分布，用于初始化，标准差等于0.01
    
    def bias_variable(shape): #进行偏置初始化
        return tf.Variable(tf.constant(0.01, shape=shape)) #进行常量初始化
    
    def conv2ds(name,x, W, b, stride=1): #设置默认的strides=1
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x,name=name)  # 使用relu激活函数
    
    def maxpool2d(name, x, k=2, stride=2):#池化操作
        return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,stride,stride,1],padding='SAME', name=name)

        
    def build_layers(self, s, c_names, w_initializer, b_initializer):
        #得到的彩色图像大小为224*320*3，需要对其进行裁剪为224*224*3，并黑白化为224*224，AlexNet上就是这个尺寸可以参考一下
        w_conv1=tf.get_variable('w_conv1',[8,8,4,32],initializer=w_initializer, collections=c_names) #卷积网络第一层的权重参数,即卷积核的大小
        b_conv1=tf.get_variable('b1', [32], initializer=b_initializer, collections=c_names)#卷积网络第一层偏置数, 这里形状不行的话就改为[1,96]
        
        w_conv2=tf.get_variable('w_conv2',[4,4,32,64],initializer=w_initializer, collections=c_names)
        b_conv2=tf.get_variable('b_conv2', [64], initializer=b_initializer, collections=c_names)
        
        
        w_conv3=tf.get_variable('w_conv3',[3,3,64,64],initializer=w_initializer, collections=c_names)
        b_conv3=tf.get_variable('b_conv3', [64], initializer=b_initializer, collections=c_names)
        
        
        w_fc1=tf.get_variable('w_fc1', [1600,512], initializer=w_initializer, collections=c_names) 
        b_fc1=tf.get_variable('b_fc1', [512], initializer=b_initializer, collections=c_names)
        
        w_fc2=tf.get_variable('w_fc2', [512,256], initializer=w_initializer, collections=c_names) 
        b_fc2=tf.get_variable('b_fc2', [256], initializer=b_initializer, collections=c_names)
        
        w_out=tf.get_variable('w_out', [256, self.n_actions], initializer=w_initializer, collections=c_names) #输出层权重
        b_out=tf.get_variable('b_out', [self.n_actions], initializer=b_initializer, collections=c_names) #输出层偏置
        
        #s=tf.placeholder(tf.float32, [None, 224, 224, 4])#定义输入
        
        #第一层卷积
        conv1_pre = tf.nn.conv2d(s, w_conv1, strides=[1, 4, 4, 1], padding='SAME')
        conv1_pre=batch_norm(conv1_pre)
        conv1_bias_pre = tf.nn.bias_add(conv1_pre, b_conv1)
        conv1=tf.nn.relu(conv1_bias_pre)
        pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        
        #把卷积与池化函数定义在类里面不知道为什么会出错，就直接写成上面卷积核池化的样子，下面同理
        #conv1=self.conv2ds('conv1', s, w_conv1, b_conv1, stride=4)
        #pool1=self.maxpool2d('pool1', conv1, k=2, stride=2)
        
        #第二层卷积
        conv2_pre= tf.nn.conv2d(pool1, w_conv2, strides=[1, 2, 2, 1], padding='SAME')
        conv2_pre=batch_norm(conv2_pre)
        conv2_bias_pre=tf.nn.bias_add(conv2_pre, b_conv2)
        conv2=tf.nn.relu(conv2_bias_pre)
        #conv2=self.conv2ds('conv2', pool1, w_conv2, b_conv2, stride=2)
        
        
        #第三层卷积
        conv3_pre= tf.nn.conv2d(conv2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
        conv3_pre=batch_norm(conv3_pre)
        conv3_bias_pre=tf.nn.bias_add(conv3_pre, b_conv3)
        conv3=tf.nn.relu(conv3_bias_pre)
        #conv3=self.conv2ds('conv3', conv2, w_conv3, b_conv3, stride=1)
        
        #全连接层1
        fc1_flat=tf.reshape(conv3, [-1, w_fc1.get_shape().as_list()[0]])
        fc1=tf.nn.relu(tf.add(tf.matmul(fc1_flat, w_fc1), b_fc1))
        fc1_drop_out=tf.nn.dropout(fc1, self.dropout)
        
        #全连接层2
        fc2_flat=tf.reshape(fc1_drop_out, [-1, w_fc2.get_shape().as_list()[0]])
        fc2=tf.nn.relu(tf.add(tf.matmul(fc2_flat, w_fc2), b_fc2))
        fc2_drop_out=tf.nn.dropout(fc2, self.dropout)
        
        #输出层
        out=tf.add(tf.matmul(fc2_drop_out, w_out), b_out)
        
        return out #输出这个状态下所有动作的Q值
    
    def builtnet(self):
        self.s=tf.placeholder(tf.float32, [None, 80, 80, 4])#定义当前这个状态的输入s
        
        '''--------------build eval_net,创建Q估计网络，输出Q估计的所有动作Q值，这个网络里面放的是最新的参数-------'''
        with tf.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer=['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()
            self.q_eval=self.build_layers(self.s, c_names, w_initializer, b_initializer)
        
        
        self.s_=tf.placeholder(tf.float32, shape=[None, 80, 80 ,4])#定义下一个状态的输入s_
        '''-------build target_net, 创建Q现实计算网络， 输出Q现实的所有动作Q值，这个网络里面放的是老的参数，若干步之后重新更新参数----'''
        with tf.variable_scope('target_net'):
            c_names=['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        
            self.q_target_pre=self.build_layers(self.s_, c_names, w_initializer, b_initializer) #创建Q现实的网络，因为Q现实和Q估计是同样的结构同样的参数
        
    
    
        self.choose_action_s=tf.placeholder(tf.float32, [None,self.n_actions]) #当前状态s下采取的动作
        Q_eval=tf.reduce_sum(tf.multiply(self.q_eval,self.choose_action_s),reduction_indices=1) #当前动作输出的Q值q_eval乘上当前的动作choose_action_s，因为当前动作choose_action_s是one_hot表示，输出的Q值Q_eval理想情况下也是one_hot表示，相乘求和就得到Q估计了
        
        self.Q_target=tf.placeholder(tf.float32, [None]) #下一状态的动作Q值，Q现实，这个是加了reward之后的Q现实
        
        cost=tf.reduce_mean(tf.square(self.Q_target-Q_eval)) #cost function就是计算当前状态值和下一状态值之间差异，让神经网络能够预测到下一状态应该怎么走
        
#        global_step = tf.Variable(0) #迭代次数初始值为0
#        #通过exponential_decay生成学习率
#        learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.96, staircase=True)
#        #0.1为初始学习率，global_step为迭代次数，100为衰减速度，0.96为衰减率
#
#        #使用指数衰减的学习率，在minimize函数中传入global_step，它将自动更新，learning_rate也随即被更新
#        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
#        #神经网络反向传播算法，使用梯度下降算法GradientDescentOptimizer来优化权重值，learning_rate为学习率，minimize中参数loss是损失函数，global_step表明了当前迭代次数(会被自动更新)
#        
        self.train_step=tf.train.AdamOptimizer(self.lr).minimize(cost) #进行训练，并指定学习率
            
        
    def store_memory(self, s_t, a_t, r_t, s_t1, terminal):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter=0
        
        self.memory.append((s_t, a_t, r_t, s_t1, terminal)) ##把当前状态s_t,当前动作：a_t, 当前带来的奖励值：r_t, s_t1:下一个状态，terminal：是否结束了，存起来
        if len(self.memory) > self.memory_size: ##如果超过这个长度了，就舍弃一点，只要memory_size多个就行
            self.memory.popleft()
        self.memory_counter+=1
         
    def image_process(self,observation): #把彩色图像处理成黑白的,observation是获得的一帧彩色的state图片
        x_t1_1=cv2.cvtColor(cv2.resize(observation, (80,80)), cv2.COLOR_BGR2GRAY)  #执行完动作之后得到的1帧图形数据，转换为80*80,并转换为灰度图
        ret, x_t1=cv2.threshold(x_t1_1, 1, 255, cv2.THRESH_BINARY) #对图像进行二值化转换，黑白两色
        
        '''构造下一个batch的数据，每4帧是一个batch，上面的的操作每次只能得到1帧图像，肯定不够，所以我们把前面
        一个batch的4帧图像的后3帧拿下来，连着这得到的1帧，就是4帧了'''
        x_t1=np.reshape(x_t1, (80,80,1)) #把上面的图像转变为80*80*1
        
        return x_t1
            
    def choose_action(self, s_t): #根据当前的状态选取动作,传进来的s_t是4帧图像重叠得到的state
        
        readout=self.sess.run(self.q_eval, feed_dict={self.s: [s_t]}) #把当前获得的4帧图像的state输入到q_eval(q估计)中，输出当前状态的动作Q值
        readout_t=readout[0]
        action_button=np.zeros([self.n_actions]) #首先将动作定义为全零的形式，要执行那个动作就在那里赋值为1，当然有可能复合动作，要改进
        action_index=0 #需要执行的动作的索引
        
        '''动作空间：
                 [[1,0,0,0,0,0], 不动
                  [0,1,0,0,0,0], 左走
                  [0,0,1,0,0,0], 右走
                  [0,0,0,0,1,0], 左翻滚
                  [0,0,0,0,0,1], 右翻滚
                  [0,0,0,1,0,0], 蹲下
        '''
        
        
        #选择动作
        '''下面选择动作的规则是，一般情况下是向前走的，向前走的动作有随机探索，有Q learning的选择最大Q值动作，
        但是在某个地方卡住的话就要返回几步，然后继续往前，判断卡住的标准是看最新的若干个reward之和是不是为0，是的话就返回几步，
        返回几步之后继续向前
        '''
        if len(self.memory) > 500  : #如果存储的记忆超过这个数目,就开始执行下面的语句
            self.last_memory.append(self.memory[len(self.memory)-1][2]) #就把存下来的记忆中最后若干个的奖励值拿出来放在一个栈里面去
    
            if len(self.last_memory) > 200: #如果栈里面的数目超过这个值了
                self.last_memory.popleft() #就删掉几个，就保留那么多就行
                last_memory_reward=sum(self.last_memory) #把后面若干个最新数据的reward拿出来求和

                if last_memory_reward <= 0 : #如果这些数据的reward之和小于等于0，就是说卡在某个画面上走不动了
                    #print('---Turn back---')
                    action_index=0 #一直卡在某个画面上，那就往回走
                    self.count_back+=1  #用这个count_back来记录返回了多少步
                    
                    if self.count_back % 18 ==0: #梅返回这么多步之后继续往前走
                        self.last_memory=deque() #并且把前面用来存最新奖励值的栈置空
                        self.count_back=0
                        self.turn_right=1 #在返回若干步之后把往前跑的标志
                    
                else:
                    #action_index=self.choose_action_index(0.2, readout_t) #选择随机动作函数
                    if np.random.random() > self.epsilon: #如果随机概率范围在探索的概率里面，就随机选择一个动作
                        #print('---Random action---')
                        #action_index=np.random.randint(self.n_actions) #选出需要执行动作的索引
                        if np.random.random() > 0.4: #//
                            action_index=1 #把随机动作设置为右跑 #//
                        else: #//
                            action_index=6 #//
                    else:
                        if np.random.random() > 0.9:
                            #print('Choose the max Q value action')
                            action_index=np.argmax(readout_t) #选择当前状态s_t下输出的Q值里，最大的Q值的索引，然后后面就选这个动作
                        else:
                            action_index=1
                   
            
            else:
                if self.turn_right > 0:
                    #print('---Turn right---')
                    action_index=1
                    self.turn_right+=1
                    if self.turn_right % 60 ==0: #返回的步数self.turn_right的大小不能超过self.last_memory，self.turn_right是返回之后向前跑的步数，self.last_memory减去self.turn_right就是返往前走之后的随机步数
                        self.turn_right=0
                else:
                    #action_index=self.choose_action_index(0.2, readout_t) #选择随机动作函数
                    if np.random.random() > self.epsilon: #如果随机概率范围在探索的概率里面，就随机选择一个动作
                        #print('---Random action---')
                        #action_index=np.random.randint(self.n_actions) #选出需要执行动作的索引
                        if np.random.random() > 0.4: #//
                            action_index=1 #把随机动作设置为右跑 #//
                        else: #//
                            action_index=6 #//
                    else:
                        if np.random.random() > 0.9:
                            #print('Choose the max Q value action')
                            action_index=np.argmax(readout_t) #选择当前状态s_t下输出的Q值里，最大的Q值的索引，然后后面就选这个动作
                        else:
                            action_index=1
                    
        else:
            #action_index=self.choose_action_index(0.2, readout_t) #选择随机动作函数
            if np.random.random() > self.epsilon: #如果随机概率范围在探索的概率里面，就随机选择一个动作
                #print('---Random action---')
                #action_index=np.random.randint(self.n_actions) #选出需要执行动作的索引
                if np.random.random() > 0.2: #//
                    action_index=1 #把随机动作设置为右跑 #//
                else: #//
                    action_index=6 #//这个是跳跃的动作索引值
            else:
                if np.random.random() > 0.9:
                    #print('Choose the max Q value action')
                    action_index=np.argmax(readout_t) #选择当前状态s_t下输出的Q值里，最大的Q值的索引，然后后面就选这个动作
                else:
                    action_index=1
                    
            self.count+=1
#            if self.count % 1000 ==0:
#                self.episode_again==False
#                self.count=0
#        
        action_button[action_index]=1 #这里把选择的那个动作索引位置1,现在还是复动作
        
        if (self.choose_act > self.observe_pic) and (self.epsilon < 0.90): #增大贪心算法的利用率，就是说逐渐增加利用率，少探索一点
            self.epsilon+=0.000001
         
#        if self.right_fast % 400 ==0:
#            self.epsilon_two=0.5*(math.sin(self.learn_step_counter)+1)+0.0001
#            self.right_fast=0
        #print(self.epsilon_two)
        
        self.choose_act+=1
            
        return action_button  #返回one-hot形式的动作选择
     
    def choose_action_index(self, choose_proba, readout_t): #这个函数是选择动作函数里面分离出来的一个选择随机动作的函数，原来的choose_action函数太大了
        #choose_proba #对某个动作的选择概率,
        #readout_t:#把当前获得的4帧图像的state输入到q_eval(q估计)中，输出当前状态的动作Q值
        
        if np.random.random() > self.epsilon: #如果随机概率范围在探索的概率里面，就随机选择一个动作
                #print('---Random action---')
                #action_index=np.random.randint(self.n_actions) #选出需要执行动作的索引
                if np.random.random() > choose_proba: #//
                    action_index=1 #把随机动作设置为右跑 #//
                else: #//
                    action_index=6 #//这个是跳跃的动作索引值
        else:
            #print('Choose the max Q value action')
            action_index=np.argmax(readout_t) #选择当前状态s_t下输出的Q值里，最大的Q值的索引，然后后面就选这个动作
        return action_index
    
    def action_button_2_onehot(self, action_button):
        #将自己定义的复合动作action_button转黄为系统的动作
        system_action_one_hot=[
                [0,0,0,0,0,0,1,0,0,0,0,0], #左跑 
                [0,0,0,0,0,0,0,1,0,0,0,0], #右跑 
                #[0,0,0,0,0,0,0,1,0,0,0,0], #右跑
                [0,0,0,0,0,1,1,0,0,0,0,0], #左翻滚
                [0,0,0,0,0,1,0,1,0,0,0,0],#右翻滚
                [0,0,0,0,0,1,0,0,0,0,0,0], #蹲下
                [0,0,0,0,0,0,0,0,0,0,0,0], #不动
                [1,0,0,0,0,0,0,0,0,0,0,0], #这个游戏中的跳跃是12位中的前两位值发生变化产生的，比如开始是[1,0,0,0,0,0,0,0,0,0,0,0]，
                [1,1,0,0,0,0,0,0,0,0,0,0],#下一次的动作是[0,0,0,0,0,0,0,0,0,0,0,0]或者[1,1,0,0,0,0,0,0,0,0,0,0]之类的就会跳跃，总之前面两位前后两次动作不一样就会跳跃
                [0,1,0,0,0,0,0,0,0,0,0,0]
                ]
        if np.argmax(action_button) > 4: #大于4是跳跃的动作索引
            a1=np.random.randint(0,2, 2)
            a0=[0,0,0,0,0,0,0,0,0,0]
            action_onehot=np.append(a1,a0)
        elif np.argmax(action_button)==1: #执行向前翻滚的动作
            a=[[0,0,0,0,0,1,0,0,0,0,0,0], #蹲下,
               [0,0,0,0,0,1,0,1,0,0,0,0]]#右跑
            if np.random.random() > 0.075: #设计向前翻滚的动作，向前翻滚是右跑和蹲下的结合
                action_ind=1
            else:
                action_ind=0
            action_onehot=a[action_ind]
        else: #执行其他的动作
            action_button_index=np.argmax(action_button)
            action_onehot=system_action_one_hot[action_button_index]
        
        return action_onehot
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action_exute):
        return self.env.step(action_exute)
    
    def _replace_target_params(self): #如何把最新的参数放到Q现实的神经网络中
        '''我们之前在建立神经网络的时候，把weight和bias等参数值都放在名字为'eval_net_params'和'target_net_params'的collection中了，
        我们要调用这些参数的时候，就直接调用对应名字的collection，要调用所有target_net_params的参数，就用tf.get_collection(),
        调用得到的是列表，把e的参数赋值到t的参数上面去'''
        t_params=tf.get_collection('target_net_params')
        e_params=tf.get_collection('eval_net_params')
        
        self.sess.run([tf.assign(t,e) for t,e in zip(t_params, e_params)])
        
    def trainNet(self):
        
        #首先判断一下是否要把target_net(Q现实)里面旧的参数替换为eval_net（Q估计）里面新的参数
        if self.learn_step_counter % self.replace_target_iter==0:
            self.sess.run(self.replace_target_op)
            #print('target_params_replaced')
        
        
        #if t > self.observe_pic: #如果存储的记忆数超过规定的数目，就开始学习，每次取出self.batch_size个
        minibatch=random.sample(self.memory, self.batch_size)
        
        s_j_batch=[d[0] for d in minibatch] #当前的状态
        
        a_batch=[d[1] for d in minibatch] #当前的动作
        
        r_batch=[d[2] for d in minibatch] #当前的奖励
            
        s_j1_batch=[d[3] for d in minibatch] #下一个状态
        #q_next表示的是在具有旧参数的Q估计网络中，用下一状态s_求得的所有动作Q值
        #q_eval4next表示的是在具有最新参数的Q估计网络中，用下一状态s_求得的所有动作Q值
        #根据double DQN的公式，下一状态的Q值选取不再是用下一状态s_输入Q现实网络(旧参数)得出所有动作Q值，然后取最大值中得出的最大值，
        #而是首先把下一状态s_输入到Q估计网络（最新参数）中得出所有动作Q值，然后选取具有最大价值的动作a；再然后，把下一状态s_输入到Q现实
        #网络中得出所有动作Q值，比如表示为Q(s)，然后把前面取得的最大动作a用到后面得到的Q值中，得到Q(s,a),这个作为下一状态价值，具体见double DQN公式   
        
        q_target=[] #Q现实值
        q_target_pre=self.sess.run(self.q_target_pre, feed_dict={self.s_: s_j1_batch}) #用Q现实网络中的老参数计算出下一状态s_的所有动作Q值
        q_eval_doubleQ=self.sess.run(self.q_eval, feed_dict={self.s: s_j1_batch})#q_eval_doubleQ表示的是在具有最新参数的Q估计网络中，用下一状态s_求得的所有动作Q值,这个是用在double DQN上面的
        
        
        #构造Q现实，就是要加上奖励和衰减因子γ的部分
        for i in range(len(minibatch)):
            terminal=minibatch[i][4] #取出minibatch里面第四维的terminal标志
            if self.double_q: #如果是double DQN算法
                if terminal: #如果已经是最后一步了，就直接把即时奖励添加到q_target里面去
                    q_target.append(r_batch[i]) 
                else:
                    select_q_index=np.argmax(q_eval_doubleQ[i]) #把q_eval_doubleQ里面最大值的索引拿出来
                    q_target.append(r_batch[i] + self.gama * q_target_pre[i][select_q_index])
            else: #普通的DQN
                if terminal: #如果已经是最后一步了，就直接把即时奖励添加到q_target里面去
                    q_target.append(r_batch[i]) 
                else:
                    q_target.append(r_batch[i] + self.gama * np.max(q_target_pre[i])) #将下一个状态s_输入到旧的网络，然后得出各个动作的Q值，从q_target_pre中选取最大的Q值，组成Q现实
                
        self.sess.run(self.train_step, feed_dict={self.Q_target:q_target, self.choose_action_s:a_batch, self.s:s_j_batch})
        
        self.learn_step_counter+=1
        #self.lr=self.lr * (0.99**(self.learn_step_counter / 1000000))
        
        #print('Learning rate: %f, epison: %f, training times: %f' %(self.lr, self.epsilon, self.choose_act))   
                

  
        
        
        
        
        