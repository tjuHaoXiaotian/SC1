import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

def test1():
    obs_n = [
        [0, 0, 1],
        [0, 2, 1],
        [1, 1, 2],
        [1, 2, 3],
    ]
    obs_n = np.array(obs_n, dtype=np.float32)
    onehot_encoder = OneHotEncoder(sparse=False)
    category_pro = onehot_encoder.fit_transform(obs_n[:, [0, 1]])
    print(category_pro)
    print(np.concatenate(np.concatenate([category_pro, obs_n[:, 2:]], axis=1)))


def test2():
    mask = tf.one_hot(
        tf.argmax([
            [1,2,3,4],
            [9,2,1,2]
                   ], axis=1), depth=4, dtype=tf.int32,
        on_value=1, off_value=0
    )

    result = tf.boolean_mask([
            [1,2,3,4],
            [9,2,1,2]
                   ], tf.cast(mask, dtype=tf.bool))

    with tf.Session() as sess:
        print(sess.run(result))

test2()