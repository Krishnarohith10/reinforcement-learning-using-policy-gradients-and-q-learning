# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:21:59 2020

@author: krish
"""

import gym
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import losses, optimizers, Model

tf.compat.v1.enable_eager_execution()

n_inputs = 4
n_hidden = 4
n_outputs = 1

X = Input(shape=(n_inputs, ))
hidden = Dense(n_hidden, activation=None)(X)
logits = Dense(n_outputs, activation=None)(hidden)
out = Activation('sigmoid')(logits)

model = Model(inputs=X, outputs=[logits, out])

p_left_right = tf.concat(axis=1, values=[out, 1-out])
action = tf.random.categorical(tf.log(p_left_right), num_samples=1)
y = 1 - tf.cast(action, tf.float32)

loss_obj = losses.BinaryCrossentropy(from_logits=True)
optimizer = optimizers.Adam(learning_rate=0.01)

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

env = gym.make('CartPole-v0')

@tf.function
def train_step(inp):
    with tf.GradientTape() as tape:
        logits, out = model(inp)
        p_left_right = tf.concat(axis=1, values=[out, 1-out])
        action = tf.random.categorical(tf.log(p_left_right), num_samples=1)
        y = 1 - tf.cast(action, tf.float32)
        loss = loss_obj(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    return action, gradients

for iteration in range(n_iterations):
    all_rewards = []
    all_gradients = []
    for game in range(10):
        current_rewards = []
        current_gradients = []
        obs = env.reset()
        for step in range(1000):
            img = env.render(mode='rgb_array')
            cv2.imshow("frames", img)
            action_val, gradients_val = train_step(obs.reshape(1, obs.shape))
            obs, reward, done, info = env.step(action_val.numpy()[0][0])
            current_rewards.append(reward)
            current_gradients.append(gradients_val)
            if done:
                cv2.destroyAllWindows()
                break
        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)
    
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=0.95)
    
        gradients = []
        for index in range(model.trainable_variables):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][index].numpy()
                                          for game_index, rewards in enumerate(all_rewards)
                                              for step, reward in enumerate(rewards)], axis=0)
            gradients.append(tf.convert_to_tensor(mean_gradients, dtype=tf.float32))
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
