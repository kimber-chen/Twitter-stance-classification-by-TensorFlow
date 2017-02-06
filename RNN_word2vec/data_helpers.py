# -*- coding: utf-8 -*-
import numpy as np
import re
import itertools
from collections import Counter
import os.path


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[()\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " exclamationmark  ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\“", " quote_f ", string)
    string = re.sub(r"\”", " quote_b ", string)
    string = re.sub(r"\?", " questionmark ", string)
    string = re.sub(r"\.", "", string)

    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    agreed_examples = list(open("data/rumour/agreed", "r").readlines())
    agreed_examples = [s.strip() for s in agreed_examples]
    disagreed_examples = list(open("data/rumour/disagreed", "r").readlines())
    disagreed_examples = [s.strip() for s in disagreed_examples]
    appeal_examples = list(open("data/rumour/appeal", "r").readlines())
    appeal_examples = [s.strip() for s in appeal_examples]
    comment_examples = list(open("data/rumour/comment", "r").readlines())
    comment_examples = [s.strip() for s in comment_examples]


    # Split by words
    x_text = agreed_examples + disagreed_examples+appeal_examples+comment_examples
    #x_text = agreed_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    agreed_labels = [[0, 0,0,1] for _ in agreed_examples]
    disagreed_labels = [[0,0,1,0] for _ in disagreed_examples]
    appeal_labels = [[0, 1,0,0] for _ in appeal_examples]
    comment_labels = [[1, 0,0,0] for _ in comment_examples]
    y = np.concatenate([agreed_labels,disagreed_labels,appeal_labels,comment_labels], 0)
    #y = np.concatenate([agreed_labels], 0)


    return [x_text, y]

def word_vec_setup ():
    if os.path.isfile("glove_twitter"): 
        word_dict = list(open("glove_twitter", "r").readlines())
    else:
        word_dict = list(open("glove_twitter_sample", "r").readlines())
        print ("------------warning-----------------")
        print ("--please get glove_twitter file--")
        print ("from http://nlp.stanford.edu/projects/glove/ ")
        print ("or vocab.npy(recommend) from https://www.dropbox.com/s/ahrz91159wtrgx1/vocab.npy?dl=1")
        print ("-----------------------------------")
    sub=[] 
    vec=[] 
    for i in xrange(len(word_dict)):#change at1113, tuple (list) to nd array
        index=word_dict[i].find(' ')
        sub.append(word_dict[i][:index])
	word=word_dict[i][index:-1].split(" ")
        #vec.append(np.array(word[1:],dtype=float))
	#print(vec)
	#print(np.array(word[1:],dtype=float))
	vec=np.append(vec,np.array(word[1:],dtype=float),axis=0)
	#vec=np.array(vec)
    	#print(vec)
    #print(type(vec))
    vec.shape=(len(word_dict),25)
    #print(vec[0])
    print("finish loading dict")
    return zip(sub,vec)
#def save_vocab(word,sub_vec):
    #if(os.path.isfile(fname)==false)
def word_vec_lookup(word,sub_vec):
    sub,vec=zip(*sub_vec)    
    for i in xrange(len(sub)):
        if (word == sub[i]):
            return vec[i]
    return np.array([0]*25)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    #data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch

        for batch_num in range(num_batches_per_epoch):
	    shuffled_data=[]
            shuffle_indices = np.random.permutation(np.arange(data_size))
	    for i in shuffle_indices:
	        shuffled_data.append(data[i])
            y_=[]
            for batch in shuffled_data:  
                #print(batch)#with vec and y_label
                x_batch, y_batch = zip(batch)   
                y_.append(y_batch)
   
            new_data=[]#new data
            a1=0
            a2=0
            a3=0
            a4=0            
            i=0

            for y in y_:


                if ((y[0][0]==1) and (a1<batch_size/4)):
                    a1+=1
                    new_data.append(shuffled_data[i])
                if ((y[0][1]==1) and (a2<batch_size/4)):
                    a2+=1
                    new_data.append(shuffled_data[i])
                if ((y[0][2]==1) and (a3<batch_size/4)):
                    a3+=1
                    new_data.append(shuffled_data[i])
                if ((y[0][3]==1) and (a4<batch_size/4)):
                    a4+=1
                    new_data.append(shuffled_data[i])                                        
                i+=1
            # print(len(new_data))
            new_data=np.array(new_data)
            indices = np.random.permutation(np.arange(len(new_data)))
            out_data = new_data[indices]            

            yield out_data
    
    #yield np.array(data)


