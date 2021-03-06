import tensorflow as tf
import numpy as np

def max_unpool_2x2(input):
    out = tf.concat([input, tf.zeros_like(input)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)
    n, h, w, c = tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]
    return tf.reshape(out, (n, h*2, w*2, c))

if __name__ == '__main__':
    test_data = tf.constant([1,2,3,4,5,6], shape=[1,2,3,1])
    output = max_unpool_2x2(test_data)
    n, h, w, c = tf.shape(test_data)[0], tf.shape(test_data)[1], tf.shape(test_data)[2], tf.shape(test_data)[3]
    sess = tf.Session()
    print(sess.run(tf.shape(test_data)))
    print(sess.run([n,h,w,c]))
    print(sess.run(tf.shape(output)))
    print(sess.run(output))
