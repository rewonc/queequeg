import tensorflow as tf
import numpy as np
import dataset
import tflib

'''
CONSTANTS
'''

BATCH_SIZE = 4
IMWIDTH = 300
IMHEIGHT = 400
TESTSPLIT = 0.20
TRIAL_NAME = 'whale1'

tr, te = dataset.get_train_test_gens(
    anno_type='Head', rel_img_path='../imgs/',
    desired_output_size=(IMWIDTH, IMHEIGHT),
    test_split_percentage=TESTSPLIT,
    annotations_dir='../code/right_whale_hunt/annotations/',
    chunk_size=BATCH_SIZE)

# input / output placeholders
x_in = tf.placeholder("float", [BATCH_SIZE, IMWIDTH, IMHEIGHT, 3])
y_in = tf.placeholder("float", [BATCH_SIZE, IMWIDTH, IMHEIGHT, 1])
# keep probability for dropout
keep_prob = tf.placeholder("float")

conv1a = tflib.conv_bn_relu(x_in,
                            kernel_size=[3, 3],
                            out_filters=32,
                            scope="conv1a",
                            summarize=True)
conv1a_do = tflib.dropout(conv1a, keep_prob)

conv1b = tflib.conv_bn_relu(conv1a_do,
                            kernel_size=[3, 3],
                            out_filters=32,
                            scope="conv1b",
                            summarize=True)
conv1b_do = tflib.dropout(conv1b, keep_prob)

conv1c = tflib.conv_bn_relu(conv1b_do,
                            kernel_size=[3, 3],
                            out_filters=32,
                            scope="conv1c",
                            summarize=True)

pool1 = tflib.max_pool_2x2(conv1c)

conv2a = tflib.conv_bn_relu(pool1,
                            kernel_size=[3, 3],
                            out_filters=64,
                            scope="conv2a",
                            summarize=True)
conv2a_do = tflib.dropout(conv2a, keep_prob)

conv2b = tflib.conv_bn_relu(conv2a_do,
                            kernel_size=[3, 3],
                            out_filters=64,
                            scope="conv2b",
                            summarize=True)
conv2b_do = tflib.dropout(conv2b, keep_prob)

conv2c = tflib.conv_bn_relu(conv2b_do,
                            kernel_size=[3, 3],
                            out_filters=64,
                            scope="conv2c",
                            summarize=True)

pool2 = tflib.max_pool_2x2(conv2c)

conv3a = tflib.conv_bn_relu(pool2,
                            kernel_size=[3, 3],
                            out_filters=128,
                            scope="conv3a",
                            summarize=True)
conv3a_do = tflib.dropout(conv3a, keep_prob)

conv3b = tflib.conv_bn_relu(conv3a_do,
                            kernel_size=[3, 3],
                            out_filters=128,
                            scope="conv3b",
                            summarize=True)
conv3b_do = tflib.dropout(conv3b, keep_prob)

conv3c = tflib.conv_bn_relu(conv3b_do,
                            kernel_size=[3, 3],
                            out_filters=128,
                            scope="conv2c",
                            summarize=True)

conv3c_do = tflib.dropout(conv3c, keep_prob)


deconv1 = tflib.deconv_bn_relu(conv3c_do,
                               kernel_size=[3, 3],
                               stride=[2, 2],
                               out_filters=64,
                               scope="deconv1",
                               summarize=True)

deconv1_do = tflib.dropout(deconv1, keep_prob)

deconv2 = tflib.deconv_bn_relu(conv3c_do,
                               kernel_size=[3, 3],
                               stride=[2, 2],
                               out_filters=32,
                               scope="deconv2",
                               summarize=True)

deconv2_do = tflib.dropout(deconv2, keep_prob)


predictions = tflib.conv_bn_relu(deconv2_do,
                                 [1, 1],
                                 1,
                                 "conv_final_1",
                                 summarize=True)

predictions_1 = tf.sigmoid(predictions)
negative_iou = tflib.negative_iou(normalized, y_in)
tf.scalar_summary("negative iou", negative_iou)
optimizer = tf.train.AdamOptimizer(2e-3).minimize(negative_iou)
summary_op = tf.merge_all_summaries()

#
# train it
#

with tr as tr_g, te as te_g:
    with tf.Session() as sess:
        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter("/tmp/" + TRIAL_NAME, sess.graph_def)
        sess.run(tf.initialize_all_variables())
        for i in xrange(20000):
            train = tr_g.next()
            sess.run(optimizer, feed_dict={x_in: train['image'], keep_prob: 0.5, y_in: train['mask'][..., np.newaxis]})
            if i % 20 == 0:
                print "batch number", i
                summary_str = sess.run(summary_op, feed_dict={x_in: train['image'], keep_prob: 0.5, y_in: train['mask'][..., np.newaxis]})
                writer.add_summary(summary_str)
            if i % 1000 == 0 and i > 0:
                test = te_g.next()
                err = sess.run(negative_iou, feed_dict={x_in: test['image'], keep_prob: 1.0, y_in: test['mask'][..., np.newaxis]})
                print 'test error: ', err
                save_path = saver.save(sess, "/tmp/whales.ckpt")
                print "Model saved in file: ", save_path
