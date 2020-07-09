import tensorflow as tf
import time

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1234)
A = tf.random.normal([10000,10000])
B = tf.random.normal([10000,10000])

def check():
    start_time = time.time()
    with tf.compat.v1.Session() as sess:
        print(sess.run(tf.reduce_sum(tf.matmul(A,B))))
    print("It took {} seconds".format(time.time() - start_time))

check()