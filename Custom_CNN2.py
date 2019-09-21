# 필요한 패키지들
import os
from glob import glob
# PIL는 이미지를 load 할 때, numpy는 array
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Hyper Parameter
batch_size = 1
data_height = 640
data_width = 640
channel_n = 4
num_classes = 3
STEPS = 1100

def get_label_from_path(path):
    return path.split('/')[-2]

def read_image(path):
    image = np.array(Image.open(path))
    # Channel 1을 살려주기 위해 reshape 해줌
    return image.reshape(image.shape[0], image.shape[1], 4)

def onehot_encode_label(path, unique_label_names):

    onehot_label = unique_label_names == get_label_from_path(path)
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label


def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

def _read_py_function(path, label):
    image = read_image(path)
    label = np.array(label, dtype=np.uint8)
    return image.astype(np.int32), label


def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded, [640, 640])
    return image_resized, label

def Data_Load():

    # label을 array 통채로 넣는게 아니고, list 화 시켜서 하나씩 넣기 위해 list로 바꿔주었다.
    train_x = glob('./TRAIN_DIR/*/*.png' )
    label_name_list = []
    for path2 in train_x:
        label_name_list.append(get_label_from_path(path2))
    unique_label_names = np.unique(label_name_list)
    train_y = [onehot_encode_label(path, unique_label_names).tolist() for path in train_x]

    #######################

    # 1. placeholder 정의
    x = tf.placeholder(tf.float32, shape=[None, 640, 640, 4])
    y_ = tf.placeholder(tf.float32, shape=[None, 3])
    is_training = tf.placeholder(tf.bool)

    ########################
    # 2. TF-Slim을 이용한 CNN 모델 구현
    with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        activation_fn=tf.nn.elu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
        inputs = tf.reshape(x, [-1, 640, 640, 4])
        net = slim.conv2d(inputs=inputs, num_outputs=100, kernel_size=[15, 15], stride=5, scope='conv1')
        net = slim.max_pool2d(inputs=net, kernel_size=[3, 3], stride=3, scope='pool1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [3, 3], stride=3, scope='pool2')

    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        activation_fn=tf.nn.elu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):

        net = slim.conv2d(net, 384, [3, 3], scope='conv3')
        net = slim.conv2d(net, 384, [3, 3], scope='conv4')
        net = slim.conv2d(net, 256, [3, 3], scope='conv5')

    with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        activation_fn=tf.nn.elu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')
        net = slim.flatten(net, scope='flatten4')

    with slim.arg_scope([slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
        net = slim.fully_connected(net, 4096, activation_fn=tf.nn.relu, scope='fc5')
        net = slim.dropout(net, is_training=is_training, scope='dropout5')
        net = slim.fully_connected(net, 2048, activation_fn=tf.nn.relu, scope='fc6')
        net = slim.dropout(net, is_training=is_training, scope='dropout6')
        net = slim.fully_connected(net, 200, activation_fn=tf.nn.relu, scope='fc7')
        outputs = slim.fully_connected(net, 3, activation_fn=None)

    ########################

    # 3. loss, optimizer, accuracy
    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=y_))
    # optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # accuracy
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ########################

    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset = dataset.map(
        lambda train_x, train_y: tuple(
            tf.py_func(_read_py_function, [train_x, train_y], [tf.int32, tf.uint8])))
    dataset = dataset.map(_resize_function)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=(int(len(train_x) * 0.4) + 3 * batch_size))
    dataset = dataset.batch(batch_size)
    #iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    # image_stacked, label_stacked = iterator.get_next()
    next_element = iterator.get_next()
    # with tf.Session() as sess:
    #     sess.run(iterator.initializer)
    #     image, label = sess.run([image_stacked, label_stacked])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 학습
        for step in range(STEPS):
            batch_xs, batch_ys = sess.run(next_element)
            _, cost_val = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs,
                                                                           y_: batch_ys,
                                                                           is_training: True})
            if (step + 1) % 500 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs,
                                                               y_: batch_ys,
                                                               is_training: False})
                print("Step : {}, cost : {:.5f}, training accuracy: {:.5f}".format(step + 1, cost_val,
                                                                                   train_accuracy))

    #     X = test_x.reshape([10, 1000, 28, 28])
    #     Y = test_y.reshape([10, 1000, 10])
    #
    #     test_accuracy = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i],
    #                                        is_training: False}) for i in range(10)])
    # print("test accuracy: {:.5f}".format(test_accuracy))






if __name__=="__main__":
    Data_Load()
