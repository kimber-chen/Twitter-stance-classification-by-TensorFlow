import numpy as np
import re
import itertools
from collections import Counter
from os import listdir
from os.path import isfile, join
import  json


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
    string = re.sub(r",", " comma", string)
    string = re.sub(r"!", " exclamationmark ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", " questionmark ", string)
    string = re.sub(r"\.\.\.", " dotdotdot ", string)
    string = re.sub(r"\.", "", string)
    return string.strip().lower()

def match_text(ID):
    json_path=""
    mypath="rumoureval-data/"

    for d in listdir(mypath):
        if ID in listdir(mypath+d):
            dir=d
            json_path=mypath+dir+"/"+ID+"/source-tweet/"+ID+".json"
            break
    
    for d in listdir(mypath):#event
        for thread in listdir(mypath+d):
            reply_path=mypath+d+"/"+thread+"/replies/"
            if ID+".json" in listdir(reply_path):
                json_path=reply_path+ID+".json"
                break
        if json_path!="":#break outer loop
            break
    if (isfile(json_path)):
        source = json.load(open(json_path, 'r'))
    else:
        return False
    return source['text']
        
def load_data_and_labels(task,jsonfile):
    """
    Loads rumorID and stance from file
    """
    ID1=[]#these list for memory ID order
    ID2=[]
    ID3=[]
    ID4=[]
    truth_values = json.load(open(jsonfile, 'r'))
    category=0
    agreed_examples=[]
    disagreed_examples=[]
    appeal_examples=[]
    comment_examples=[]
    if (task=="taskA"):
        category=4#SDQC
        for id in truth_values.keys():
            if(match_text(id)):
                if (truth_values[id]=="support"):
                    agreed_examples.append(match_text(id))
                    ID1.append(id)
                if (truth_values[id]=="deny"):
                    disagreed_examples.append(match_text(id))
                    ID2.append(id)
                if (truth_values[id]=="query"):
                    appeal_examples.append(match_text(id))
                    ID3.append(id)
                if (truth_values[id]=="comment"):
                    comment_examples.append(match_text(id))  
                    ID4.append(id)              
    else:
        category=3#rumor or not    

    #strip special char
    agreed_examples = [s.strip() for s in agreed_examples]
    disagreed_examples = [s.strip() for s in disagreed_examples]
    appeal_examples = [s.strip() for s in appeal_examples]
    comment_examples = [s.strip() for s in comment_examples]

    IDlist=ID1+ID2+ID3+ID4
    # Split by words
    x_text = agreed_examples + disagreed_examples+appeal_examples+comment_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    agreed_labels = [[0, 0,0,1] for _ in agreed_examples]
    disagreed_labels = [[0,0,1,0] for _ in disagreed_examples]
    appeal_labels = [[0, 1,0,0] for _ in appeal_examples]
    comment_labels = [[1, 0,0,0] for _ in comment_examples]
    y = np.concatenate([agreed_labels,disagreed_labels,appeal_labels,comment_labels], 0)



    return [x_text, y,IDlist]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch

        for batch_num in range(num_batches_per_epoch):
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            y_=[]
            for batch in shuffled_data:  
                #print(batch)
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


