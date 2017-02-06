#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import data_helpers
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
import sys
import operator
import  json

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("stop_step", 1500, "stop at (default: 1500)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Task parameters
tf.flags.DEFINE_string("task", "taskA", "semeval2017 TaskA")
tf.flags.DEFINE_string("training_file", "traindev/rumoureval-subtaskA-train.json", "file define training data set and annotation")
tf.flags.DEFINE_string("testing_file", "traindev/rumoureval-subtaskA-dev.json", "file define training data set and annotation (not use)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

task=FLAGS.task
training_file=FLAGS.training_file
testing_file=FLAGS.testing_file
# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
global testID
x_train, y_train,trainID = data_helpers.load_data_and_labels(task,training_file)
x_dev,y_dev,testID= data_helpers.load_data_and_labels(task,testing_file)
  
# Build vocabulary

x_text=x_train+x_dev
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

#Split train/test set
x_train=x[:len(x_train)]
x_dev=x[len(x_train):]

# Randomly shuffle data
'''np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
'''


# Training
# ==================================================
print("start training...")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=4,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        
        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step,  loss, accuracy = sess.run(
                [train_op, global_step,  cnn.loss, cnn.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def write_json(scores):
            f=open("result.json","w")
            output_dict={}
            for i in xrange(len(testID)):
                m=max(scores[i])
                position=[a for a, j in enumerate(scores[i]) if j == m]
                if type(position)==list:
                    position=position[0]

                if (position==0):
                    category="comment"
                if (position==1):
                    category="query"
                if (position==2):
                    category="deny"
                if (position==3):
                    category="support"
                output_dict[str(testID[i])]=category
            f.write(json.dumps(output_dict,separators=(',', ': ')))

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
          
            fo=open("result.txt","w")
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step,  loss, accuracy,scores = sess.run(
                [global_step,  cnn.loss, cnn.accuracy,cnn.scores],
                feed_dict)
            y_label=[]
            for i in y_batch:
                for x in xrange(4):
                    if(i[x]==1):
                         y_label.append(x)

            for ind in xrange(len(scores)):
                fo.write (str(scores[ind])+","+str(y_label[ind])+"\n")
            fo.write("\n")
            time_str = datetime.datetime.now().isoformat()
            print("eval{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            write_json(scores)


        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
	#print batch
            x_batch, y_batch = zip(*batch)  
	#print x_batch
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                #print("\nEvaluation:")
                dev_step(x_dev, y_dev)
            if(current_step==FLAGS.stop_step):
                break
           
