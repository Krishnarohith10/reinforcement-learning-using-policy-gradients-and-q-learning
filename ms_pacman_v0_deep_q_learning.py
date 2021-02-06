import gym
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.keras import Model, initializers

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], enable=True)

mspacman_color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.mean(axis=2)
    img[img==mspacman_color] = 0
    img = (img - 128)/128 - 1
    return img.reshape(88, 80, 1)

env = gym.make('MsPacman-v0')

input_height = 88
input_width = 80
input_channels = 1
conv_filters = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ['same']*3
conv_activation = ['relu']*3
n_hidden = 512
n_outputs = env.action_space.n

initializer = initializers.VarianceScaling()

prev_layer = Input(shape=(88,80,1))
input_layer = prev_layer

for filters, kernel_size, strides, padding, activation in zip(
        conv_filters, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
    prev_layer = Conv2D(filters=filters, kernel_size=kernel_size,
                     strides=strides, padding=padding, activation=activation, 
                     kernel_initializer = initializer)(prev_layer)
flatten_out = Flatten()(prev_layer)
dense_out = Dense(n_hidden, activation='relu', kernel_initializer=initializer)(flatten_out)
outputs = Dense(n_outputs, activation='relu', kernel_initializer=initializer)(dense_out)

actor_q_network = Model(inputs=input_layer, outputs=outputs)
critic_q_network = Model(inputs=input_layer, outputs=outputs)

replay_memory_size = 500000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, done
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return np.argmax(q_values)

n_steps = 4000000 # total number of training steps
training_start = 10000 # start training after 1,000 game iterations
training_interval = 4 # run a training step every 3 game iterations
copy_steps = 10000 # copy the critic to the actor every 25 training steps
discount_rate = 0.99
skip_start = 90 # skip the start of every game (it's just waiting time)
batch_size = 50
iteration = 0 # game iterations
done = True # env needs to be reset
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.001, 
                                                 momentum=0.95, use_nesterov=True)

@tf.function
def train_step(X_state_val, X_action_val, y_val, n_outputs):
    with tf.GradientTape() as tape:
        actor_q_val = actor_q_network(X_state_val)
        q_val = tf.reduce_sum(actor_q_val * tf.one_hot(X_action_val, n_outputs), 
                              axis=1, keepdims=True)
        error = tf.abs(y_val - q_val)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
        
    gradients = tape.gradient(loss, actor_q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, actor_q_network.trainable_variables))
    
    return loss

step = 0

while True:
    if step >= n_steps:
        break
    iteration += 1
    print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
        iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
    if done:
        obs = env.reset()
        for skip in range(skip_start):
            obs, reward, done, info = env.step(0)
        state = preprocess_observation(obs)
    
    q_val = actor_q_network(tf.expand_dims(state, axis=0))
    action = epsilon_greedy(q_val, step)
    
    #Actor plays
    obs, reward, done, info = env.step(action)
    next_state = preprocess_observation(obs)
    
    replay_memory.append((state, action, reward, next_state, 1.0 - done))
    state = next_state
    
    #Compute statistics for tracking process
    total_max_q += np.max(q_val)
    game_length += 1
    if done:
        mean_max_q = total_max_q/game_length
        game_length = 0
        total_max_q = 0
    
    if iteration < training_start or iteration % training_interval != 0:
        continue
    
    # Critic learns
    X_state_val, X_action_val, rewards, X_next_state_val, continues = (
        sample_memories(batch_size))
    next_q_values = critic_q_network(X_next_state_val)
    max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
    y_val = rewards + continues * discount_rate * max_next_q_values
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    
    loss_val = train_step(X_state_val, X_action_val, y_val, n_outputs)
    
    if step % copy_steps == 0:
        critic_q_network.set_weights(actor_q_network.get_weights())
    
    step += 1

#Save trained model
actor_q_network.save('actor_q_network_new.h5')

