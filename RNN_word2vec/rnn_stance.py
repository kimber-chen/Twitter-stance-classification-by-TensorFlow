from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import data_helpers
from tensorflow.contrib import learn
import os

# Parameters
# ==================================================


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 400, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("stop_step", 400, "model training times")
tf.flags.DEFINE_string("event", "sydneysiege", "event name (default: sydneysiege)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

event=FLAGS.event
fo=open(event+"_rst.txt","w")

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_text, y_label = data_helpers.load_data_and_labels()

#class_list=["ottawashooting","charliehebdo","ferguson","sydneysiege"]

my_list=[]

for x in xrange(len(x_text)):
    i=x_text[x].find('*')
    sub=x_text[x][:i]
    x_text[x]=x_text[x][i+1:]
    if(sub==event):
        i1=x_text[x].find('*')
        sub1=x_text[x][:i1]#threadID      
        #x_text[x]=x_text[x][i1:]      
        my_list.append(sub1)

my_list = list(set(my_list)) #threadID list  
test_num=[]
for rum in my_list:            
    temp=[]    
    for x in xrange(len(x_text)):
        i1=x_text[x].find('*')
        if (i1!=-1):
            if(rum==x_text[x][:i1]):
                temp.append(x)
                #x_text[x]=x_text[x][i1:]
    test_num.append(temp)

#data sub string
for x in xrange(len(x_text)):
    x_text[x]=x_text[x][19:]

#----------------------------------loading if exist  
outfile="vocab"
max_document_length = max([len(x.split(" ")) for x in x_text]) # Build vocabulary
if os.path.isfile(outfile+".npy"):
    x_=np.load(outfile+".npy")
else:
   
    #Return word2vec
    sub_vec=data_helpers.word_vec_setup()
    x_vec=[]
    for k in x_text:
        k_sp=k.split(" ")#each word in instance
        sent_vec=[]
        i=-1
        for _ in xrange(len(k_sp)):
            i+=1    
            sent_vec=np.append(sent_vec,data_helpers.word_vec_lookup(k_sp[i],sub_vec))
        for _ in xrange(i+1,max_document_length):
            sent_vec=np.append(sent_vec,np.array([0]*25))        
        x_vec=np.append(x_vec,sent_vec)
    x_vec.shape=(-1,max_document_length,25)
    np.save(outfile, x_vec)#save vec to file
    x_=x_vec
#--------------------------------------------------


# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = FLAGS.batch_size
display_step = 10

# Network Parameters
n_input = 25 # MNIST data input (img shape: 28*28)
n_steps = max_document_length # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 4 # MNIST total classes (0-9 digits)

model_n=0
for n in test_num:

    model_n+=1
    x_train,y_train=x_,y_label
    x_dev,y_dev=[],[]
    inplace=0

    for num in n:
        x_dev.append(x_[num])
        y_dev.append(y_label[num])
        x_train = np.delete(x_train, num-inplace,0)
        y_train = np.delete(y_train, num-inplace,0) 
        inplace+=1
        #print(x_text[num])
        #print(x_[num])

    #print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


        # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    def print_full(arr):
        print ("len(arr):"+str(len(arr)))
        print ("len(arr[0]):"+str(len(arr[0])))
        aa=""
        for i in xrange(len(arr)):
            for j in xrange(len(arr[0])):
                aa+=str(arr[i][j])+" "
            print (aa)
            print ("------------------------")

    def RNN(x, weights, biases,model_n):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)
	with tf.variable_scope("myrnn", reuse = None) as scope:
            # Define a lstm cell with tensorflow
	    if (model_n>1):
	        scope.reuse_variables()#not to share variable
            lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)
            # Get lstm cell output
            outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred = RNN(x, weights, biases,model_n)


    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
	fo=open(event+"_rst.txt","a")
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        #while step * batch_size < training_iters:
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        for batch in batches:
		
	#for i in xrange(1000):
            x_batch, y_batch = zip(*batch)  
            #x_batch = x_train.reshape((batch_size, n_steps, n_input))#warning ori is x_batch.reshape
	    x_batch=np.array(x_batch)
	    y_batch=np.array(y_batch)
           #print_full(batch_x[0])
              # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: x_batch, y: y_batch})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy 

        pred,accuracy=sess.run([pred,accuracy],feed_dict={x: x_dev, y: y_dev})
        print(pred)
        for i in xrange(len(pred)):
	    if (y_dev[i][0]==1):
                fo.write (str(pred[i])+", 0\n")
	    elif (y_dev[i][1]==1):
                fo.write (str(pred[i])+", 1\n")
	    elif (y_dev[i][2]==1):
                fo.write (str(pred[i])+", 2\n")
	    elif (y_dev[i][3]==1):
                fo.write (str(pred[i])+", 3\n")
        fo.write("\n")

