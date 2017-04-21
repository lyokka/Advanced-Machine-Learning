import os.path

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
    dat = np.zeros((batch_size, seq_length, shape, shape, 3))
    for i in xrange(batch_size):
        dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
    return dat 


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

fourcc = cv2.cv.CV_FOURCC('m', 'j', 'p', 'g')


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
loss = tf.nn.l2_loss(x[:,FLAGS.seq_start+1:,:,:,:] - pred[:,FLAGS.seq_start:,:,:,:])

# define a training operation
train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)   

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
    _, loss_r = sess.run([train_op, loss],
                         feed_dict={x: data,
                                    keep_prob: FLAGS.keep_prob})

    # print loss for troubleshooting... should be decreasing in trend
    if step % 10 == 0:
        print('======= step number {} ======='.format(step))
        print('loss: {}'.format(loss_r))

    assert not np.isnan(loss_r), 'Model diverged with loss = NaN'


# -------------------------------------------------------------------------------
# generate dataset
import skvideo.io


for i in xrange(5000):
	# after training is complete, generate a new batch of data
	data = generate_bouncing_ball_sample(FLAGS.batch_size,
	                                     FLAGS.seq_length,
	                                     32,
	                                     FLAGS.num_balls)

	# make videos
	
	print('now generating the generated video {} !'.format(i))
	video = cv2.VideoWriter()
	ims = sess.run([pred], feed_dict={x: data, keep_prob: 1.0})
	ims = ims[0][0, FLAGS.seq_start-1:, :, :]
	ims = np.maximum(ims, 0)*255
	ims = ims.astype(np.uint8)
	skvideo.io.vwrite('VideoDataSet/g{}.avi'.format(i), ims)
	print(ims.shape)

	print('now generating the true video {} !'.format(i))
	video = cv2.VideoWriter()
	ims = data[0, FLAGS.seq_start:, :, :, :]
	ims = np.maximum(ims, 0)*255
	ims = ims.astype(np.uint8)
	skvideo.io.vwrite('VideoDataSet/t{}.avi'.format(i), ims)
	print(ims.shape)
	video.release()
