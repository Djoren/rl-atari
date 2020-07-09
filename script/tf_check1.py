import numpy as np
import tensorflow as tf
from datetime import datetime

tf.compat.v1.disable_eager_execution()

# Choose which device you want to test on: either 'cpu' or 'gpu'
devices = ['cpu', 'gpu']

# Choose size of the matrix to be used.
# Make it bigger to see bigger benefits of parallel computation
shapes = [(50, 50), (100, 100), (500, 500), (1000, 1000), (20000, 10000)]


def compute_operations(device, shape):
    """Run a simple set of operations on a matrix of given shape on given device

    Parameters
    ----------
    device : the type of device to use, either 'cpu' or 'gpu' 
    shape : a tuple for the shape of a 2d tensor, e.g. (10, 10)

    Returns
    -------
    out : results of the operations as the time taken
    """

    # Define operations to be computed on selected device
    with tf.device(device):
        random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
        dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        sum_operation = tf.reduce_sum(dot_operation)

    # Time the actual runtime of the operations
    start_time = datetime.now()
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
            result = session.run(sum_operation)
    elapsed_time = datetime.now() - start_time
    print(elapsed_time)


compute_operations(devices[0], shapes[-1])