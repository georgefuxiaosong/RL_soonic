# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:07:08 2018

@author: Fuxiao
"""

#-------------------主体程序----------------------------
import retro
from RL_brain import DeepQNetwork
import numpy as np
import tensorflow as tf
import cv2
from collections import deque
            
def playgame():
    
    env=retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    
    total_steps=0#总的步数
    
    
    sess=tf.Session()
    RL=DeepQNetwork(n_actions=9, double_q=False, sess=sess) #初始化类
    '''
    设计到复合动作，比如右走同时跳，但是gym的retro环境是一个env.action_space.sample()长度为12d的one_hot数组，每个位置上的1表示一个单一
    的动作，比如[0,0,0,0,0,0,1,0,0,0,0,0]表示只往右走，为了获得复合动作，重新设计如下动作，并对设计的复合动作和retro的默认动作进行转换
                  [[1,0,0,0,0,0,0], 不动
                  [0,1,0,0,0,0,0], 左走
                  [0,0,1,0,0,0,0], 右走
                  [0,0,0,0,1,0,0], 左翻滚
                  [0,0,0,0,0,1,0], 右翻滚
                  [0,0,0,1,0,0,0], 蹲下
                  [0,0,0,0,0,0,1]
    在输入进行训练的时候，动作空间就只有这6种，找到对应动作后再进行转换
    '''
    
    sess.run(tf.global_variables_initializer()) #初始化所有变量,变量的初始化必须在模型的其它操作运行之前先明确地完成。最简单的方法就是添加一个给所有变量初始化的操作，并在使用模型之前首先运行那个操作
   
    saver=tf.train.Saver() #保存训练完的所有变量
    checkpoint=tf.train.get_checkpoint_state('saved_networks') #把训练过模型的放到saved_networks这个文件夹下，下次可以接着从这里开始训练
                
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path) #如果以前训练过这个模型，可以接着训练
        print('Successfully loaded: ', checkpoint.model_checkpoint_path)
    else:
        print('loaded failed') #如果没有这个模型就从头开始训练'''
    
    for episode in range(2000):
        action_stack=deque() #用来记录当前动作需要执行重复次数的栈
        info_rings=deque() #用来记录info中rings的变换量的栈
        info_rings.append(0) #先放一个0进去
        
        #total_reward=0 #总的奖励值
        observation=env.reset() #初始化环境，获得最开始的observation，这个observation是一幅彩色图像,但这一帧彩色图像暂时不用
        
        #observation, reward, done, _=env.step(env.action_space.sample())#随机执行一个动作
        
        '''回合最开始的时候没有4帧图像，就用最初的图像重叠4帧'''
        #x_t_1, r_0, terminal, _=env.step(env.action_space.sample()) #随机执行一个动作，执行完一帧之后得到的图像数据x_t, reward:r_0, terminal:是否终止
        x_t_2=cv2.cvtColor(cv2.resize(observation, (80,80)), cv2.COLOR_BGR2GRAY) #执行完一帧之后得到的图形数据，转换为80*80,并转换为灰度图
        ret, x_t=cv2.threshold(x_t_2, 1, 255, cv2.THRESH_BINARY) #对图像进行二值化转换，黑白两色
           
        s_t=np.stack((x_t,x_t,x_t,x_t),axis=2) #开始的时候只有一帧图形，batch需要4帧作为一个state，我们把这一帧叠在一起,成为80*80*4的结构
        #x=cv2.cvtColor(cv2.resize(observation_, (80,80)), cv2.COLOR_BGR2YCR_CB)
        
        while True:
#            env.render() #渲染环境
            
            #下面的动作选择，由于是每4帧图像重叠在一起作为一个state，那么我就把这4帧图像的动作设置成一样，意思就是每个动作重复执行4次
            
            if len(action_stack) > 0: #如果存放动作的栈不空，那么就从里面选择一个动作来执行
                action_exute=action_stack.pop() #随意出栈4个动作的其中一个
            else: #如果动作栈里面的 动作用完了，那么再次用state选出一个动作
                action_button=RL.choose_action(s_t) #根据这4帧图像选择动作,这个动作是自己设计的复合动作
#                a=[[0,0,0,0,0,1,0,0,0,0,0,0], #蹲下,
#                   [0,0,0,0,0,1,0,1,0,0,0,0]]#右跑
#                if np.random.random() > 0.2: #设计向前翻滚的动作，向前翻滚是右跑和蹲下的结合
#                    action_ind=1
#                else:
#                    action_ind=0
#                action_system=a[action_ind]
                action_system=RL.action_button_2_onehot(action_button)#np.append(np.append(a0,a1),b1)# #转变为系统所能是被的动作
                for i in range(3):
                    action_stack.append(action_system) #把这个动作放到栈里面去，放3个，加上自己的一个，就是4个重复动作
                action_exute=action_system
                
            
            #action_exute=action_stack.pop()
            
            observation_, reward, done, info=env.step(action_exute) #用当前动作获得下一帧的observation_， 当前的奖励等等信息
            info_rings.append(info['rings']) #把执行当前动作获得的info里面的rings的个数存到info_rings的栈里面去
            
            if info_rings[1] -info_rings[0] < 0: #如果后面的一次动作使得rings的数量减少了，那么奖励值就要减少
                reward-=50
            elif info_rings[1] - info_rings[0] > 0: #如果后面的一次动作使得rings的数量增加了了，那么奖励值就要增加
                reward+=50
            else:
                pass
            
            info_rings.popleft()
            
            #total_reward+=reward
            #print(total_reward)
            
            #让他往右边走，所以在往右边走的这个动作上设计一个比较小的奖励
            #if np.argmax(action_system)==7:
             #   reward=0.1
            
            if done: #如果中途死掉了，就给一个负的奖励
                reward=-100
            
            x_t1_pre=cv2.cvtColor(cv2.resize(observation_, (80,80)), cv2.COLOR_BGR2GRAY) #执行完一帧之后得到的图形数据，转换为80*80,并转换为灰度图
            ret, x_t1=cv2.threshold(x_t1_pre, 1, 255, cv2.THRESH_BINARY) #对图像进行二值化转换，黑白两色
            x_t1=np.reshape(x_t1,(80,80,1)) #把上面执行动作得到的一帧图像，转换变成80*80*1，
            
            #x_t1=RL.image_process(observation_[0]) #把上面执行完动作之后得到的一帧彩色图像observation_处理成黑白的80*80*1的大小
            s_t1=np.append(s_t[:, :, :3], x_t1, axis=2) #把前面3帧和计算得到的1帧加起来作为下一个state
            
            
            #把当前的状态s_t，当前采取的动作action,当前奖励reward，下一个状态s_t1，是否结束的标志存起来
            #需要注意的是当前的状态s_t和下一个状态s_t1是4帧图像叠起来的，所以需要在前面进行处理
            
            RL.store_memory(s_t, action_button, reward, s_t1, done) #存的时候还是存自己设计的复合动作按键
            
            if (total_steps > 3000) and (total_steps%80==0): #在步数超过200补之后开始学习，并且每50步学习一次
                with tf.device("/gpu:0"):
                    RL.trainNet() #开始训练网络，学习
               
            if done:
                break
            
            s_t=s_t1
            
            total_steps+=1
            
            if total_steps % 50000==0:
               saver.save(sess, './', global_step=total_steps)
                    
            '''state=''
            if total_steps < RL.observe_pic:
                state='Observe'
            else:
                state='Train'
                print("TIMESTEP", RL.learn_step_counter, "/ STATE", state, \
                      "/ EPSILON", RL.epsilon,  "/ REWARD", observation, \
                      "/ Q_MAX %e" % np.max(action))
                '''
            
    print('Game over')
    env.close()
    

if __name__=='__main__':
    playgame()
