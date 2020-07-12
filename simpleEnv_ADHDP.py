import gym
import gym_carai
import time
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # this line forces to run on CPU
import tensorflow as tf

# print(tf.config.experimental.list_physical_devices('GPU'))  # show all gpus
# print(tf.__version__)  # show tf version
tf.keras.backend.set_floatx('float32')

# C / A neurons
# 80 / 50 works ~ 150 runs Adam  learning_rate = 0.0001
# 30 / 30 does not ~ 2+K Adam learning_rate = 0.0001
# 50 / 30 - works, Adam LR + 0.0001,  488 at worst
# 50 / 10 - works, Adam LR + 0.0001, 750 at worst
# 50 / 5  - Nah
# 40 / 10 - Nope

# TODO: Reporting
# TODO: Advanced env

def map_to_range(x, min_out=-1, max_out=1):
    # custom
    x_out = tf.keras.backend.tanh(x)  # x in range(-1,1)
    scale = (max_out-min_out)/2.
    min_out += 1
    return  x_out * scale + min_out


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.JStar = []

    def store(self, state, action, reward, JOpt):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.JStar.append(JOpt)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.JStar = []


class CriticModel(tf.keras.Model):
    def __init__(self, learning_rate, observation_shape):
        super(CriticModel, self).__init__()

        neurons_inner_layer = 40

        # critic part of model (value function)
        self.dense1 = tf.keras.layers.Dense(neurons_inner_layer, activation='relu')
        self.value = tf.keras.layers.Dense(observation_shape) # condense back into 2

        self.opt = tf.keras.optimizers.Adam(learning_rate)

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=2)
        x = self.dense1(inputs)
        J = self.value(x)
        return J


class ActorModel(tf.keras.Model):
    def __init__(self, learning_rate, observation_shape):
        super(ActorModel, self).__init__()
        self.observation_shape = observation_shape
        neurons_inner_layer = 10

        # actor part of Model (policies)
        self.dense1 = tf.keras.layers.Dense(neurons_inner_layer, activation='relu')
        self.turning = tf.keras.layers.Dense(1, activation=map_to_range)  # sigmoid for turning direction

        self.opt = tf.keras.optimizers.Adam(learning_rate)

    def call(self, inputs):
        x = self.dense1(inputs)
        dir = self.turning(x)
        return dir


def get_loss_critic(Critic, memory, gamma=0.99):
    # get value for each timestep
    values = Critic(tf.convert_to_tensor(np.vstack(memory.actions), dtype=tf.float32))

    # gamma is the forgetting factor, prioritize short term rewards over long term.
    reward_sum = 0
    discounted_rewards = []
    for reward in memory.rewards[::-1]:
        reward_sum = reward + gamma * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards = discounted_rewards[::-1]
    # get J for each timestep
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values
    critic_loss = advantage ** 2
    critic_loss = tf.reduce_mean(critic_loss)
    return critic_loss, discounted_rewards


def get_loss_actor(Actor, Critic, memory):
    # get value for each timestep, based on the reward derived from the critic.
    # get critic reward through actor response to allow gradienttape to get the differences.
    values = Critic(Actor(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32)))
    # get J for each timestep
    err = values - tf.convert_to_tensor(np.array(memory.JStar)[:, None], dtype=tf.float32)
    actor_loss = err ** 2
    actor_loss = tf.reduce_mean(actor_loss)
    return actor_loss


# Setup for training etc.
env = gym.make('carai-simple-v0')  # First open the environment
observation_shape = env.observation_space.shape[0]

# load some settings and the models
maxEpoch = 100000                  # max amount of epochs
maxEpochTime = 120                 # [s] max seconds to spend per epoch
dt = 1/60                          # fps (should equal monitor refresh rate)
maxSteps = int(maxEpochTime/dt)    # max duration of an epoch
Terminate = None
learning_rate = 0.0001
load_model = input('Load_model? Y/N \n')

if load_model == "Y" or load_model == "y":
    model_name = input("Model Folder Name \n")
    actor_load_name = 'Models/'+model_name+'/Actor'
    critic_load_name = 'Models/'+model_name+'/Critic'
    Actor = tf.keras.models.load_model(actor_load_name)
    Critic = tf.keras.models.load_model(critic_load_name)
    Train = False
else:
    Actor = ActorModel(learning_rate, observation_shape)  # global network
    Critic = CriticModel(learning_rate, observation_shape)  # global network
    Train = True

# some initial values
done = 0
mem = Memory()
maxRewardSoFar = -90000
corresponding_critic = -90000
reward_average_list = []
critic_average_list = []
start_time = time.time()           # Register current time
epoch = 1                          # Current episode
while epoch < maxEpoch:
    print("--- starting run %s ---" % epoch)
    run_time = time.time()
    env.reset()
    mem.clear()

    # initial values
    epoch_loss = 0
    action = np.array([0])
    obs, reward, done, info_dict, Terminate = env.step(action, dt)
    i = 0
    while i < maxSteps:
        # calculate next step
        env.render('human')  # manual, human, human-vsync, rgb_array
        action = Actor(obs).numpy()[0]  # returns 0,1,2, action space = -1 to 1
        obs, reward, done, info_dict, Terminate = env.step(action, dt)
        mem.store(obs, action[0], reward, info_dict['JStar'])
        if Terminate:  # Window was closed.
            epoch = maxEpoch*2
            i = maxSteps * 2
            run = False
        if done:
            break
        i += 1
    if Train:
        # GradientTape tracks the gradient of all variables within scope, useful for optimizer
        with tf.GradientTape() as critic_tape:
            critic_loss, critic_rewards = get_loss_critic(Critic, mem)
        # Apply found gradients to model
        critic_grads = critic_tape.gradient(critic_loss, Critic.trainable_weights)
        Critic.opt.apply_gradients(zip(critic_grads, Critic.trainable_weights))

        with tf.GradientTape() as actor_tape:
            actor_loss = get_loss_actor(Actor, Critic, mem)
        actor_grads = actor_tape.gradient(actor_loss, Actor.trainable_weights)
        Actor.opt.apply_gradients(zip(actor_grads, Actor.trainable_weights))

        print("--- Actor loss {} - ".format(actor_loss))
        print("--- Critic loss {} - ".format(critic_loss))

        # Statistics for training progress evaluation
        reward_average = sum(mem.rewards)/len(mem.rewards)
        critic_average = sum(critic_rewards)/len(critic_rewards)
        reward_average_list.append(reward_average)
        critic_average_list.append(critic_average)
        if reward_average > maxRewardSoFar:
            maxRewardSoFar = reward_average
            corresponding_critic = critic_average
        print("- Highest reward yet {} - {} - ".format(maxRewardSoFar, corresponding_critic))
        print("- Most recent run reward {} - {} - ".format(reward_average, critic_average))
        if len(reward_average_list) > 11:
            last = reward_average_list[-10:]
            last2 = critic_average_list[-10:]
            print("- Last ten runs average {}, - {} - ".format(sum(last)/len(last), sum(last2)/len(last2)))
    print("--- %s seconds ---" % (time.time() - run_time))

    epoch += 1
env.close()
print("--- total %s seconds ---" % (time.time() - start_time))


save = input('save model? Y/N \n')
if save == "Y" or save == "y":
    model_save_name = input('Model folder name \n')
    actor_save_name = 'Models/'+model_save_name+'/Actor'
    critic_save_name = 'Models/'+model_save_name+'/Critic'

    tf.keras.models.save_model(Actor, actor_save_name)
    tf.keras.models.save_model(Critic, critic_save_name)
