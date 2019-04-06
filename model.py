#!/usr/bin/python
# -*- coding: utf-8 -*-

# External library modules
import tensorflow as tf
import numpy as np

# library modules
import operator
import functools
import os
import time

# local modules
from data import LSVRC2010
from logs import get_logger
from utils import read_vgg_conf

class VGG:
    """
    This is the tensorflow implementation of vgg.

    This is the tensorflow implementation of
    `Very Deep Convolutional Networks for Large Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    def __init__(self, path):
        """
        Build the VGG model
        """
        self.vgg_conf = read_vgg_conf()
        width, height = self.vgg_conf['input_size']
        self.input_images = tf.placeholder(tf.float32,
                                           shape=[None, width, height, 3],
                                           name='input_image')
        self.output_labels = tf.placeholder(tf.float32,
                                            shape=[None, self.vgg_conf['FC19']],
                                            name='output_labels')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        self.global_step = tf.Variable(tf.constant(0))

        self.path = path
        self.model_path = os.path.join(os.getcwd(), 'model', 'model.ckpt')
        if not os.path.exists(os.path.join(os.getcwd(), 'model')):
            os.mkdir(os.path.join(os.getcwd(), 'model'))

        self.logger = get_logger()

    def get_weight(self, layer):
        """
        Initilize the weights of layer with xavier initializer

        Initialize the weights of each convolutional
        layer as explained `here <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

        :param layer: layer number for which to create weight for
        """
        initializer = tf.contrib.layers.xavier_initializer()
        _shape = [self.vgg_conf['filter'], self.vgg_conf['filter'],
                  self.vgg_conf['conv' + str(layer - 1)],
                  self.vgg_conf['conv' + str(layer)]]

        w = tf.Variable(initializer(shape=_shape),
                        name='weight' + str(layer))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)

        return w

    def get_strides(self):
        """
        Return the strides for a tf.nn.conv2d.
        """
        return [1, self.vgg_conf['stride'], self.vgg_conf['stride'], 1]

    def vgg_conv(self, _input, layer):
        """
        Convolutional layer in vgg.

        The convolutional layer is initialized with
        xavier(glorot) initializer with stride of 1
        and 'SAME' padding.

        :param _input: input to the convolutional layer
        :param layer: layer number of the convolutional layer.
        """
        return tf.nn.conv2d(_input,
                            self.get_weight(layer),
                            strides=self.get_strides(),
                            padding=self.vgg_conf['padding'])

    def vgg_conv_test(self, _input, layer, shape):
        """
        Fully connected equivalent convolutional layer in vgg.

        Here the convolutional layer is initialized with
        xavier(glorot) initializer with stride of 1
        and 'SAME' padding. But the layer weight will be loaded
        from the trained model.

        :param _input: input to the convolutional layer
        :param layer: layer number of the convolutional layer.
        :param shape: shape of the weight matrix
        """
        initializer = tf.contrib.layers.xavier_initializer()
        if layer == 17:
            shape_l1 = self.vgg_conf['conv' + str(layer - 1)]
        else:
            shape_l1 = self.vgg_conf['FC' + str(layer - 1)]
        _shape = [shape[0], shape[1],
                  shape_l1, self.vgg_conf['FC' + str(layer)]]

        w = tf.Variable(initializer(shape=_shape),
                        name='weight' + str(layer))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)

        return tf.nn.conv2d(_input,
                            w, strides=self.get_strides(),
                            padding=self.vgg_conf['padding'])

    def vgg_pooling(self, _input):
        """
        Max pooling layer in VGG.

        The max pooling layer of vgg is of window size
        (2 x 2) and stride of 2.

        :param _input: input to the max pooling layer.
        """
        return tf.layers.max_pooling2d(_input,
                                       [self.vgg_conf['max-pooling']['filter'],
                                        self.vgg_conf['max-pooling']['filter']],
                                       self.vgg_conf['max-pooling']['stride'])

    def vgg_fully_connected(self, _input, layer):
        """
        Create a fully connected layer for vgg.

        Manually create the layer instead of using
        tf.contrib.layers.fully_connected so it will be easier to
        change it to convolutional layer during test time.

        :param _input: input to the fully connected layer
        :param layer: layer number of the fully connected layer
        """
        input_shape = _input.get_shape().as_list()[1:]
        weight_shape = [functools.reduce(operator.mul, input_shape),
                        self.vgg_conf['FC' + str(layer)]]

        weight = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=weight_shape),
                             name='weight' + str(layer))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
        bias = tf.Variable(tf.constant(0.0, shape=[self.vgg_conf['FC' + str(layer)]]),
                           name='bias' + str(layer))

        return tf.matmul(_input, weight) + bias

    @property
    def l2_loss(self):
        """
        Calculate the l2 loss of the vgg graph
        """
        lambd = 5e-4
        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
        l2 = lambd * tf.reduce_sum([tf.nn.l2_loss(weight) for weight in weights])
        return l2

    def build_graph_conv_1_to_16(self, input_images):
        """
        Build the graph for first 16 layers

        First 16 layers are convolutional layers. They will
        be same for training as well as inference.
        """
        with tf.name_scope('group1'):
            with tf.variable_scope('conv1'):
                conv1 = self.vgg_conv(input_images, 1)
                conv1 = tf.nn.relu(conv1)

            with tf.variable_scope('conv2'):
                conv2 = self.vgg_conv(conv1, 2)
                conv2 = tf.nn.relu(conv2)

        with tf.variable_scope('maxpool1'):
            maxpool1 = self.vgg_pooling(conv2)

        with tf.name_scope('group2'):
            with tf.variable_scope('conv3'):
                conv3 = self.vgg_conv(maxpool1, 3)
                conv3 = tf.nn.relu(conv3)

            with tf.variable_scope('conv4'):
                conv4 = self.vgg_conv(conv3, 4)
                conv4 = tf.nn.relu(conv4)

        with tf.variable_scope('maxpool2'):
            maxpool2 = self.vgg_pooling(conv4)

        with tf.name_scope('group3'):
            with tf.variable_scope('conv5'):
                conv5 = self.vgg_conv(maxpool2, 5)
                conv5 = tf.nn.relu(conv5)

            with tf.variable_scope('conv6'):
                conv6 = self.vgg_conv(conv5, 6)
                conv6 = tf.nn.relu(conv6)

            with tf.variable_scope('conv7'):
                conv7 = self.vgg_conv(conv6, 7)
                conv7 = tf.nn.relu(conv7)

            with tf.variable_scope('conv8'):
                conv8 = self.vgg_conv(conv7, 8)
                conv8 = tf.nn.relu(conv8)

        with tf.variable_scope('maxpool3'):
            maxpool3 = self.vgg_pooling(conv8)

        with tf.name_scope('group4'):
            with tf.variable_scope('conv9'):
                conv9 = self.vgg_conv(maxpool3, 9)
                conv9 = tf.nn.relu(conv9)

            with tf.variable_scope('conv10'):
                conv10 = self.vgg_conv(conv9, 10)
                conv10 = tf.nn.relu(conv10)

            with tf.variable_scope('conv11'):
                conv11 = self.vgg_conv(conv10, 11)
                conv11 = tf.nn.relu(conv11)

            with tf.variable_scope('conv12'):
                conv12 = self.vgg_conv(conv11, 12)
                conv12 = tf.nn.relu(conv12)

        with tf.variable_scope('maxpool4'):
            maxpool4 = self.vgg_pooling(conv12)

        with tf.name_scope('group5'):
            with tf.variable_scope('conv13'):
                conv13 = self.vgg_conv(maxpool4, 13)
                conv13 = tf.nn.relu(conv13)

            with tf.variable_scope('conv14'):
                conv14 = self.vgg_conv(conv13, 14)
                conv14 = tf.nn.relu(conv14)

            with tf.variable_scope('conv15'):
                conv15 = self.vgg_conv(conv14, 15)
                conv15 = tf.nn.relu(conv15)

            with tf.variable_scope('conv16'):
                conv16 = self.vgg_conv(conv15, 16)
                conv16 = tf.nn.relu(conv16)

        with tf.variable_scope('maxpool5'):
            maxpool5 = self.vgg_pooling(conv16)

        return maxpool5

    def build_graph(self):
        """
        Creates tensorflow graph for VGG before training.

        It has 19 layers
            -> 16 convolutional layers
            -> 3 fully connected layers.
            -> 5 max pooling layers

        The rough architecture is as follows
        input_image (224 x 224 x 3) -> conv1 (224 x 224 x 64)
        -> conv2 (224 x 224 x 64) -> maxpooling (112 x 112 x 64)
        -> conv3 (112 x 112 x 128) -> conv4 (112 x 112 x 128)
        -> maxpooling (56 x 56 x 128) -> conv5 (56 x 56 x 256)
        -> conv6 (56 x 56 x 256) -> conv7 (56 x 56 x 256)
        -> conv8 (56 x 56 x 256) -> maxpooling (28 x 28 x 256)
        -> conv9 (28 x 28 x 512) -> conv10 (28 x 28 x 512)
        -> conv11 (28 x 28 x 512) -> conv12 (28 x 28 x 512)
        -> maxpooling (14 x 14 x 512) -> conv13 (14 x 14 x 512)
        -> conv14 (14 x 14 x 512) -> conv15 (14 x 14 x 512)
        -> conv16 (14 x 14 x 512) -> maxpooling (7 x 7 x 512)
        == FC (25088) -> FC (4096) -> FC (4096) -> FC (1000)
        """
        maxpool5 = self.build_graph_conv_1_to_16(self.input_images)

        flatten = tf.layers.flatten(maxpool5)

        # Fully connected layers

        with tf.variable_scope('FC17'):
            fc17 = self.vgg_fully_connected(flatten, 17)
            fc17 = tf.nn.relu(fc17)
            fc17 = tf.nn.dropout(fc17, keep_prob=self.dropout)

        with tf.variable_scope('FC18'):
            fc18 = self.vgg_fully_connected(fc17, 18)
            fc18 = tf.nn.relu(fc18)
            fc18 = tf.nn.dropout(fc18, keep_prob=self.dropout)

        with tf.variable_scope('FC19'):
            fc19 = self.vgg_fully_connected(fc18, 19)

        # Loss
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=fc19,
            labels=self.output_labels
        )
        self.loss = tf.reduce_mean(self.loss) + self.l2_loss

        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).\
                                                    minimize(self.loss,
                                                             global_step=self.global_step)

        # Accuracies
        equal = tf.equal(tf.argmax(fc19, 1),
                         tf.argmax(self.output_labels, 1))
        self.top1 = tf.reduce_mean(tf.cast(equal, tf.float32))

        top5 = tf.nn.in_top_k(predictions=fc19,
                              targets=tf.argmax(self.output_labels, 1), k=5)
        self.top5 = tf.reduce_mean(tf.cast(top5, tf.float32))

        self.add_summaries()

    def train(self, epochs, batch_size=256, learning_rate=1e-2, restore=True):
        """
        Train the vgg graph

        Train the VGG network by batch by batch and gather summaries
        in tensorboard to visualize loss and accuracies.

        :param epochs: number of epochs to run for the training
        :param batch_size: batch size of the model while training
        """
        self.build_graph()

        lsvrc2010 = LSVRC2010(self.path, batch_size)
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            train_summary_writer, val_summary_writer = self.get_summary_writers(sess)

            if restore and os.path.exists(os.path.abspath(os.path.join(self.model_path, '..'))):
                saver.restore(sess, self.model_path)
                self.logger.info("Model Restored from path: %s",
                                 self.model_path)
                step = sess.run(self.global_step)
                if step > 155000:
                    learning_rate /= 10
                if step > 300000:
                    learning_rate /= 10
                if step > 490000:
                    learning_rate /= 10
            else:
                sess.run(init)

            for _ in range(epochs):
                next_batch = lsvrc2010.gen_batch
                start = time.time()
                for images, labels in next_batch:
                    feed_dict = {self.input_images: images,
                                 self.output_labels: labels,
                                 self.learning_rate: learning_rate,
                                 self.dropout: 0.5}
                    summaries, _, step = sess.run([self.merged_summaries,
                                                   self.optimizer,
                                                   self.global_step],
                                                  feed_dict=feed_dict)

                    if step == 155000:
                        learning_rate /= 10
                        self.logger.info("Learning rate is decreased to %f", learning_rate)
                    if step == 300000:
                        learning_rate /= 10
                        self.logger.info("Learning rate is decreased to %f", learning_rate)
                    if step == 490000:
                        learning_rate /= 10
                        self.logger.info("Learning rate is decreased to %f", learning_rate)

                    train_summary_writer.add_summary(summaries, step)

                    if step % 10 == 0:
                        end = time.time()
                        feed_dict[self.dropout] = 1.0
                        loss, top1, top5 = sess.run([self.loss,
                                                     self.top1, self.top5],
                                                    feed_dict=feed_dict)

                        self.logger.info("10 batches took (in seconds): %f | Training | "
                                         "Step: %d Loss: %f Top1: %f Top5: %f",
                                         end - start, step, loss, top1, top5)
                        start = time.time()

                    if step % 100 == 0:
                        save_path = saver.save(sess, self.model_path)
                        self.logger.info("Model saved in path: %s", save_path)

                    if step % 500 == 0:
                        val_images, val_labels = lsvrc2010.get_batch_val
                        feed_dict = {self.input_images: val_images,
                                     self.output_labels: val_labels,
                                     self.dropout: 1.0}
                        (val_summaries, loss, top1,
                         top5) = sess.run([self.merged_summaries,
                                           self.loss, self.top1, self.top5],
                                          feed_dict=feed_dict)
                        val_summary_writer.add_summary(val_summaries, step)

                        self.logger.info("Validation | Step: %d Loss: %f Top1: %f Top5: %f",
                                         step, loss, top1, top5)

    def get_graph_weights(self):
        """
        Return all the weight numbers from stored graph

        For each layer store the weights in a dictionary
        and return it.
        """
        self.build_graph()
        saver = tf.train.Saver()
        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
        graph_weights = {}
        with tf.Session() as sess:
            if restore and os.path.exists(os.path.abspath(os.path.join(self.model_path, '..'))):
                saver.restore(sess, self.model_path)
            for weight in weights:
                graph_weights[weight.name.split(':')[0]] = sess.run(weight)
        tf.reset_default_graph()
        return graph_weights

    def build_graph_test(self):
        """
        Build the test graph.

        Convert all fully connected layers into equivalent
        convolutional layers. Keep all other configurations
        as same as training graph
        """
        self.input_images_test = tf.placeholder(tf.float32,
                                                shape=[None, None, None, 3],
                                                name='input_image_test')
        self.output_labels_test = tf.placeholder(tf.float32,
                                                 shape=[None, self.vgg_conf['FC19']],
                                                 name='output_labels_test')

        maxpool5 = self.build_graph_conv_1_to_16(self.input_images_test)

        # Fully connected layers
        with tf.variable_scope('FC17'):
            fc17 = self.vgg_conv_test(maxpool5, 17, [7, 7])
            fc17 = tf.nn.relu(fc17)

        with tf.variable_scope('FC18'):
            fc18 = self.vgg_conv_test(fc17, 18, [1, 1])
            fc18 = tf.nn.relu(fc18)

        with tf.variable_scope('FC19'):
            fc19 = self.vgg_conv_test(fc18, 19, [1, 1])

        self.fc19_avg = tf.reduce_mean(fc19, [1, 2])

        # Loss
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.fc19_avg,
            labels=self.output_labels_test
        )
        self.loss = tf.reduce_mean(self.loss) + self.l2_loss

    def assign_weight(self, sess, graph_weights):
        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)

        for weight in weights:
            weight_values = graph_weights[weight.name.split(':')[0]]
            layer = int(weight.name.split(':')[0].split('/')[-1][6:])
            if layer == 17:
                weight_values = weight_values.reshape([7, 7,
                                                       self.vgg_conf['conv' + str(layer - 1)],
                                                       self.vgg_conf['FC' + str(layer)]])
            if layer in [18, 19]:
                weight_values = weight_values.reshape([1, 1,
                                                       self.vgg_conf['FC' + str(layer - 1)],
                                                       self.vgg_conf['FC' + str(layer)]])
            sess.run(tf.assign(weight, weight_values))

    def test(self, batch_size=64):
        """
        Test the vgg graph.

        :param batch_size: batch size of the model while training
        """
        graph_weights = self.get_graph_weights()
        self.build_graph_test()

        lsvrc2010 = LSVRC2010(self.path, batch_size)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            self.assign_weight(sess, graph_weights)

            next_batch = lsvrc2010.gen_batch_test
            avg_loss, avg_top1, avg_top5 = [], [], []
            start = time.time()
            for idx, (images, labels) in enumerate(next_batch):
                loss, fc19 = 0, 0
                for i in range(3):
                    feed_dict = {self.input_images_test: images[i],
                                 self.output_labels_test: labels}
                    _loss, _fc19 = sess.run([self.loss, self.fc19_avg],
                                            feed_dict=feed_dict)
                    loss += _loss
                    fc19 += _fc19

                loss /= 3
                fc19 /= 3

                top1 = np.argmax(fc19) == np.argmax(labels)
                top5 = np.argmax(labels) in np.argpartition(_fc19, -5)[0][-5:]

                avg_loss.append(loss)
                avg_top1.append(top1)
                avg_top5.append(top5)

                self.logger.info("Testing | Image No: %d Loss: %f Top1: %f Top5: %f",
                                 idx,
                                 sum(avg_loss) / len(avg_loss),
                                 sum(avg_top1) / len(avg_top1),
                                 sum(avg_top5) / len(avg_top5))
                if (idx + 1) % 100 == 0:
                    end = time.time()
                    self.logger.info("It took %f seconds for 100 images", end - start)
                    start = time.time()

    ########################  TENSORBOARD  ############################

    def add_summaries(self):
        """
        Add loss, top1 and top5 summaries to visualize.
        """
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("top1-accuracy", self.top1)
        tf.summary.scalar("top5-accuracy", self.top5)

        self.merged_summaries = tf.summary.merge_all()

    def get_summary_writers(self, sess):
        """
        Returns summary writers for train and validation summary
        """
        train_summary = tf.summary.FileWriter(
            os.path.join(os.getcwd(), 'graph', 'train'), sess.graph)

        val_summary = tf.summary.FileWriter(
            os.path.join(os.getcwd(), 'graph', 'val'), sess.graph)

        return train_summary, val_summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar = 'image-path',
                        help = 'ImageNet dataset path')
    parser.add_argument('--restore_model', metavar = 'restore-model', default='true',
                        help = 'true if you want to restore the saved model')
    parser.add_argument('--train', metavar = 'train', default='true',
                        help = 'true if you want to train the model')
    parser.add_argument('--test', metavar = 'test', default='true',
                        help = 'true if you want to test the model')
    args = parser.parse_args()

    restore = True if args.restore_model == 'true' else False
    vgg = VGG(args.image_path)
    if args.train == 'true':
        vgg.train(50, batch_size=64, restore=restore)
    if args.test == 'true':
        vgg.test()
