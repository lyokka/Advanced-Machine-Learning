import tensorflow as tf
import numpy as np
import skvideo.io

# load a sample
vid = skvideo.io.vread('VideoDataSet/g0.avi')
vid = vid.reshape((1,)+vid.shape)

# load data set
traX = np.load('VideoDataNpy/vid_tra.npy')
traY = np.load('VideoDataNpy/vid_tra_label.npy')
tesX = np.load('VideoDataNpy/vid_tes.npy')
tesY = np.load('VideoDataNpy/vid_tes_label.npy')

# initialize weight
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# X: input data with size [None, 5, 32, 32, 3]
# w: weight for conv1
# w2: weight for conv2
# w3: weight for conv3
# w4: weight for fully_connected1
# w_o: weight for output_layer
# p_c: percentage of droput for conv
# p_h: percentage of droput for fully_connected
def model(X, weights, biases, p_c, p_h):
    conv1 = tf.nn.conv3d(X, weights['conv1'], strides=[1,1,1,1,1], padding='SAME')
    l1 = tf.nn.relu(conv1+biases['conv1'])
    l1 = tf.nn.max_pool3d(l1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_c)

    conv2 = tf.nn.conv3d(l1, weights['conv2'], strides=[1,1,1,1,1], padding='SAME')
    l2 = tf.nn.relu(conv2+biases['conv2'])
    l2 = tf.nn.max_pool3d(l2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_c)

    conv3 = tf.nn.conv3d(l2, weights['conv3'], strides=[1,1,1,1,1], padding='SAME')
    l3 = tf.nn.relu(conv3+biases['conv3'])
    l3 = tf.nn.max_pool3d(l3, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_c)
    l3 = tf.reshape(l3, [-1, weights['fc'].get_shape().as_list()[0]])    

    l4 = tf.nn.relu(tf.matmul(l3, weights['fc'])+biases['fc'])
    l4 = tf.nn.dropout(l4, p_h)

    output = tf.matmul(l4, weights['out']) + biases['out']

    return output

#--------------------------------------------------------------------
X = tf.placeholder('float', [None, 5, 32, 32, 3])
Y = tf.placeholder('float', [None, 2])


weights = {
    'conv1': init_weight([3, 3, 3, 3, 10]),
    'conv2': init_weight([3, 3, 3, 10, 20]),
    'conv3': init_weight([3, 3, 3, 20, 40]),
    'fc': init_weight([5*128, 625]),
    'out': init_weight([625, 2])
    }

biases = {
    'conv1': init_weight([10]),
    'conv2': init_weight([20]),
    'conv3': init_weight([40]),
    'fc': init_weight([625]),
    'out': init_weight([2])
}


p_c = tf.placeholder('float')
p_h = tf.placeholder('float')
batch_size = tf.placeholder('int32')

Y_ = model(X, weights, biases, p_c, p_h)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_, labels=Y))
optimizer = tf.train.AdamOptimizer(0.003).minimize(loss)

pred = tf.argmax(Y_, 1)
truth = tf.argmax(Y, 1)
correct_pred = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

display_step = 20 * 128

#--------------------------------------------------------------------
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10):
        for start, end in zip(range(0, len(traX), 128), range(128, len(traX)+1, 128)):
            _, cost, acc = sess.run([optimizer, loss, accuracy],
                                  feed_dict={X:traX[start:end], Y:traY[start:end],
                                       p_c:0.8, p_h:0.5})

            if start % display_step == 0:
                print "Epoch:{}".format(i) + "   Step:{}".format(start) + "    acc:{}".format(acc)

    print "Testing Acc:{}".format(sess.run(accuracy, feed_dict={X:tesX, Y:tesY, p_c:1, p_h:1}))
    #Prediction, Truth = sess.run([pred, truth], feed_dict={X:tesX, Y:tesY, p_c:1, p_h:1})
    # print result ----------------------------------------------------------------------------
    #for i, j in zip(Prediction, Truth):
    #    print 'pred:{}'.format(i) + '   truth:{}'.format(j)
