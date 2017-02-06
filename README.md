#Tensorflow Twitter stances classification
Use Tensorflow to run CNN, and RNN model. And add word2vec to represent the text.

The original data can be found here:
[PHEME rumour scheme dataset: journalism use case, version 2](https://figshare.com/articles/PHEME_rumour_scheme_dataset_journalism_use_case/2068650)

##1.CNN_word embedding

Requrement(Optional)
  * Download full dataset "rumoureval-data"
  * RUN:
  `python train.py`

Method from [IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

My code is for competition [RumourEval-2017, subtask A](https://competitions.codalab.org/competitions/16171)

##2.RNN_word2vec


Requrement(Optional)
  * Download [vocab.npy](https://www.dropbox.com/s/ahrz91159wtrgx1/vocab.npy?dl=1)
  * OR  Downolad GloVe to build the vocab.npy data. This will take the program for about an hour.
  * RUN:
  `python rnn_stance.py`

GloVe for word2vec expression.
LOO (Leave One Out) to evaluate the result. That is, use only one conversion thread as testing data. The rest data as training data. 

Then we compare with other learning method on different rumor events. Performance especial good for F1 measure.
![alt tag](https://www.dropbox.com/s/zf372v2i5qygeo2/Twitter_stance.PNG?raw=1)
Reference: M. Lukasik, P. K. Srijith, D. Vu, K. Bontcheva, A. Zubiaga, T. Cohn. Hawkes Processes for Continuous Time Sequence Classification: an Application to Rumour Stance Classification in Twitter. ACL. 2016.)
