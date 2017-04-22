import os.path

import numpy as np
import tensorflow as tf
import skvideo.io

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell

# define flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_directory',
                           """ directory to store trained model""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 5,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                            """for dropout""")
tf.app.flags.DEFINE_float('learning_rate', .003,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
tf.app.flags.DEFINE_bool('friction', False,
                            """whether there is friction in the system""")
tf.app.flags.DEFINE_integer('num_balls', 2,
                            """num of balls in the simulation""")
#---------------------------------------------------------------------------------------------------
# define the generator graph
def generator(x_dropout):

    # configure the generator graph
    def conv_lstm_network(inputs, hidden):
        
        # some convolutional layers before the convLSTM cell
        conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1")
        conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
        conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3")
        conv4 = ld.conv_layer(conv3, 1, 1, 4, "encode_4")

        # take output from first 4 convolutional layers as input to convLSTM cell
        with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell.BasicConvLSTMCell([8, 8], [3, 3], 4)
            if hidden is None:
                hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
            cell_output, hidden = cell(conv4, hidden)

        # some convolutional layers after the convLSTM cell
        conv5 = ld.transpose_conv_layer(cell_output, 1, 1, 8, "decode_5")
        conv6 = ld.transpose_conv_layer(conv5, 3, 2, 8, "decode_6")
        conv7 = ld.transpose_conv_layer(conv6, 3, 1, 8, "decode_7")

        # the last convolutional layer will use linear activations
        x_1 = ld.transpose_conv_layer(conv7, 3, 2, 3, "decode_8", True)

        # return the output of the last conv layer, and the hidden cell state
        return x_1, hidden

    # make a template for variable reuse
    network = tf.make_template('network', conv_lstm_network)

    # cell outputs will be stored here
    x_unwrap = []

    # initialize hidden state to None in first cell
    hidden = None

    # loop over each frame in the sequence, sending through convLSTM cells
    for i in xrange(FLAGS.seq_length-1):

        # look at true frames for the first 'seq_start' samples
        if i < FLAGS.seq_start:
            x_1, hidden = network(x_dropout[:, i, :, :, :], hidden)

        # after 'seq_start' samples, begin making predictions and propagating
        # through LSTM network
        else:
            x_1, hidden = network(x_1, hidden)

        # add outputs to list
        x_unwrap.append(x_1)

    # stack and reorder predictions
    x_unwrap = tf.stack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])

    # return the prediction tensor
    return x_unwrap
#-------------------------------------------------------------------------------------------------
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
def discrimator_network(D_X, weights, biases, p_c, p_h):
    conv1 = tf.nn.conv3d(D_X, weights['conv1'], strides=[1,1,1,1,1], padding='SAME')
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

D_X = tf.placeholder('float', [None, 5, 32, 32, 3])
D_Y = tf.placeholder('float', [None, 2])


D_weights = {
    'conv1': init_weight([3, 3, 3, 3, 10]),
    'conv2': init_weight([3, 3, 3, 10, 20]),
    'conv3': init_weight([3, 3, 3, 20, 40]),
    'fc': init_weight([5*128, 625]),
    'out': init_weight([625, 2])
}

D_biases = {
    'conv1': init_weight([10]),
    'conv2': init_weight([20]),
    'conv3': init_weight([40]),
    'fc': init_weight([625]),
    'out': init_weight([2])
}

D_p_c = tf.placeholder('float')
D_p_h = tf.placeholder('float')

D_Y_ = discrimator_network(D_X, D_weights, D_biases, D_p_c, D_p_h)
D_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_Y_, labels=D_Y))
D_op = tf.train.AdamOptimizer(0.003).minimize(D_loss)

D_pred = tf.argmax(D_Y_, 1)
D_truth = tf.argmax(D_Y, 1)
D_correct_pred = tf.equal(tf.argmax(D_Y_, 1), tf.argmax(D_Y, 1))
D_accuracy = tf.reduce_mean(tf.cast(D_correct_pred, tf.float32))
    
#-------------------------------------------------------------------------------------------------
def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
    dat = np.zeros((batch_size, seq_length, shape, shape, 3))
    for i in xrange(batch_size):
        dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
    return dat 

# configure input parameters... stored as flags
if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
tf.gfile.MakeDirs(FLAGS.train_dir)

# make input tensor, and wrap it with dropout
x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 32, 32, 3])
keep_prob = tf.placeholder("float")
x_dropout = tf.nn.dropout(x, keep_prob)

# define a predictor operation
pred = generator(x_dropout)

# define loss operation
# this is the L2 loss between true and predicted frames... we'll change this
# to an adveserial loss when we add the descriminator
G_loss = tf.nn.l2_loss(x[:,FLAGS.seq_start+1:,:,:,:] - pred[:,FLAGS.seq_start:,:,:,:])

# define a training operation
G_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(G_loss)   

# define an initialization operation
init_op = tf.global_variables_initializer()

# run operations on the graph
# with tf.Session() as sess:
sess = tf.Session() # only using this method for debugging.. should use above

print('building network')
sess.run(init_op)

# run specified number of training steps
for step in xrange(FLAGS.max_step):
    
    # generate a batch of training data
    data = generate_bouncing_ball_sample(FLAGS.batch_size,
                                         FLAGS.seq_length,
                                         32,
                                         FLAGS.num_balls)
    
    # run training and loss operations on the graph
    _, G_loss_r, D_input= sess.run([G_op, G_loss, pred],
                                   feed_dict={x: data,
                                              keep_prob: FLAGS.keep_prob})

    # discrimator step
    if (step % 10 == 0) & (step != 0):
        print 'discrimator start'
        d_vid = np.empty((300, 5, 32, 32, 3))
        d_vid_label = np.zeros((300, 2))
        
        # generate data
        for i in range(0, 300, 2):
            # after training is complete, generate a new batch of data
	    d_data = generate_bouncing_ball_sample(FLAGS.batch_size,
	                                           FLAGS.seq_length,
	                                           32,
	                                           FLAGS.num_balls)
            d_gen = sess.run([pred], feed_dict={x: d_data, keep_prob: 1.0})
            d_gen = d_gen[0][0, FLAGS.seq_start-1:, :, :]
	    d_gen = np.maximum(d_gen, 0)*255

            d_true = d_data[0, FLAGS.seq_start:, :, :, :]
	    d_true = np.maximum(d_true, 0)*255

            d_vid[i,:,:,:,:] = d_gen
            d_vid[i+1,:,:,:,:] = d_true
            d_vid_label[i, 0] = 1
            d_vid_label[i+1, 1] = 1

        # shuffle data
        ind = np.arange(300)
        d_vid = d_vid[ind]
        d_vid_label = d_vid_label[ind]

        # train discimator
        _, cost = sess.run([D_op, D_loss], feed_dict={D_X:d_vid,
                                                           D_Y:d_vid_label,
                                                           D_p_c:0.8,
                                                           D_p_h:0.5})
        print '\t D_cost:{}'.format(cost)
        
        
        print 'discrimator end'
    
    # print loss for troubleshooting... should be decreasing in trend
    if step % 10 == 0:
        print('======= step number {} ======='.format(step))
        print('loss: {}'.format(G_loss_r))

    assert not np.isnan(G_loss_r), 'Model diverged with loss = NaN'
