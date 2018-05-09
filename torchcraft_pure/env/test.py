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
            [1, 2, 3, 4],
            [9, 2, 1, 2]
        ], axis=1), depth=4, dtype=tf.int32,
        on_value=1, off_value=0
    )

    result = tf.boolean_mask([
        [1, 2, 3, 4],
        [9, 2, 1, 2]
    ], tf.cast(mask, dtype=tf.bool))

    with tf.Session() as sess:
        print(sess.run(result))


def test3():
    array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # mask = [[True, False, False, False]]
    mask = np.array([[0, 0, 0, 0], [0, 0, 0, 1]])
    mask_result = tf.boolean_mask(array, mask)  # N x action_dim
    with tf.Session() as sess:
        print(sess.run(mask_result))

def test4():
    array = [
        np.array([[1,],[2,],[3,]]),
        np.array([[4,],[5,],[6,]])
        ]
    # result = np.concatenate(array, axis=1)
    # print(result)
    result = tf.concat(array, axis=1)
    with tf.Session() as sess:
        print(sess.run(result))
    #     print(tf.reduce_sum(np.array(array), axis=0, keep_dims=False).shape[0])
    #     print(sess.run(tf.reshape(tf.reduce_sum(np.array(array), axis=0, keep_dims=False), shape=[4, 1])))

def test5():
    print(np.argmax([0,0,0,0]))


def test6():
    # dead = tf.placeholder(dtype=tf.bool, shape=[5,1])
    dead_p = tf.placeholder(dtype=tf.bool, shape=[5])
    x_p = tf.placeholder(dtype=tf.int32, shape=[5,2])
    y_p = tf.placeholder(dtype=tf.int32, shape=[5,2])
    dead = [True,False,True,True,False]
    x = [[1, 2],[2, 3],[3, 4],[4, 5],[5, 6]]
    y = [[-1, -1],[-1, -1],[-1, -1],[-1, -1],[-1, -1]]
    result = tf.where(dead_p, x_p, y_p)
    with tf.Session() as sess:
        print(sess.run(result, feed_dict={
            dead_p: dead,
            x_p: x,
            y_p: y
        }))
test6()
