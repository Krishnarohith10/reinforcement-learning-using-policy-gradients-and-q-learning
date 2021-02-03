import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import losses, optimizers, Model

gpu=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], enable=True)

n_inputs = 4
n_hidden = 4
n_outputs = 1

X = Input(shape=(n_inputs, ))
hidden = Dense(n_hidden, activation=None)(X)
logits = Dense(n_outputs, activation=None)(hidden)
out = Activation('sigmoid')(logits)

model = Model(inputs=X, outputs=[logits, out])

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

loss_obj = losses.BinaryCrossentropy(from_logits=True)
optimizer = optimizers.Adam(learning_rate=0.01)

@tf.function
def train_step(inp):
    with tf.GradientTape() as tape:
        logits, out = model(inp)
        p_left_right = tf.concat(axis=1, values=[out, 1-out])
        action = tf.random.categorical(tf.math.log(p_left_right), num_samples=1)
        y = 1 - tf.cast(action, tf.float32)
        loss = loss_obj(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    return action, gradients

n_iterations = 450

for iteration in range(n_iterations):
    print("\rIteration: {}".format(iteration), end="")
    all_rewards = []
    all_gradients = []
    for game in range(10):
        current_rewards = []
        current_gradients = []
        obs = env.reset()
        for step in range(1000):
            img = env.render(mode='rgb_array')
            action_val, gradients_val = train_step(obs.reshape(1, obs.shape[0]))
            obs, reward, done, info = env.step(action_val.numpy()[0][0])
            current_rewards.append(reward)
            current_gradients.append(gradients_val)
            if done:
                break
        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)
    
    # At this point we have run the policy for 10 episodes, and we are
    # ready for a policy update using the algorithm described earlier.
    all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=0.95)
    
    gradients = []
    for index in range(len(model.trainable_variables)):
        # multiply the gradients by the action scores, and compute the mean
        mean_gradients = np.mean([reward * all_gradients[game_index][step][index].numpy() 
                                  for game_index, rewards in enumerate(all_rewards)
                                  for step, reward in enumerate(rewards)], axis=0)
        gradients.append(tf.convert_to_tensor(mean_gradients, dtype=tf.float32))
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

model.save('cart_pole_v0_using_policy_gradients.h5')
env.close()

