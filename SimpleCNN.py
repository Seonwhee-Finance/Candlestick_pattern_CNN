import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def my_cnn(images, num_classes, is_training):  # is_training is not used...
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        net = slim.fully_connected(net, num_classes, activation_fn=None)
        return net


def make_tensor_graph():
    data_path = './OHLC_train.tfrecords'

    with tf.Graph().as_default():
        batch_size, height, width, channels = 10, 640, 640, 3
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['train/image'], tf.float32)
        label = tf.cast(features['train/label'], tf.int32)
        image = tf.reshape(image, [640, 640, 3])
        images, labels = tf.train.shuffle_batch([image, label], batch_size, capacity=30, num_threads=1,
                                                min_after_dequeue=10)

        num_classes = 3
        logits = my_cnn(images, num_classes, is_training=True)  ########################
        probabilities = tf.nn.softmax(logits)

        with tf.Session() as sess:
            init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
            # Run the init_op, evaluate the model outputs and print the results:
            sess.run(init_op)
            probabilities = sess.run(probabilities)
            print(probabilities.shape)

            # Create a coordinator, launch the queue runner threads.
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # try:
            #     while not coord.should_stop():
            #         while True:
            #             prob = sess.run(probabilities)
            #             print('Probabilities Shape:')
            #             print(prob.shape)
            #
            # except tf.errors.OutOfRangeError:
            #     # When done, ask the threads to stop.
            #     print('Done training -- epoch limit reached')
            # finally:
            #     coord.request_stop()
            #     # Wait for threads to finish.
            # coord.join(threads)

            # Save the model
            saver = tf.train.Saver()
            saver.save(sess, './slim_model/custom_model')



if __name__ == '__main__':

    make_tensor_graph()
