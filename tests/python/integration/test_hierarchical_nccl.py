import tensorflow as tf

from kungfu.tensorflow.ops.collective import _hierarchical_nccl_all_reduce


def test_with_sizes(sizes, n_steps=10):
    assert (len(sizes) == 1)
    xs = [tf.Variable(tf.ones((n, ), dtype=tf.int32)) for n in sizes]
    ys = [_hierarchical_nccl_all_reduce(x) for x in xs]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(n_steps):
            vs = sess.run(ys)


def main():
    test_with_sizes([1024])


main()
