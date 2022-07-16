import gfootball.env as football_env
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

env = football_env.create_environment(env_name='11_vs_11_hard_stochastic', representation='simple115v2', render=True, rewards='checkpoints,scoring')

# env = football_env.create_environment(env_name='11_vs_11_easy_stochastic', representation='pixels', render=True, rewards='checkpoints,scoring')


# new_actor = tf.keras.models.load_model('./model_sinucigas_imageBased/model_actor.hdf5')
# new_actor = tf.keras.models.load_model('./model_ucigas/model_actor_ucigas.hdf5')
# new_actor = tf.keras.models.load_model('./model_actor_last.hdf5')
# new_actor = tf.keras.models.load_model('./paseRapide_imageBased/model_actor_1.hdf5')
new_actor = tf.keras.models.load_model('./model_actor_last.hdf5')

n_actions = env.action_space.n
dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))


def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        action_probs = new_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 1000:
            break
    return total_reward


test_reward()