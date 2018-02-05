'''
Created on 2017年3月23日

@author: zry
'''

import tensorflow as tf
impfrom ntm_test import train as dnc_train

class Dnc:
    def __init__(self, data_dir=None):
        # Parameters
        self.learning_rate = 0.001
        self.batch_size = 200
        # Network Parameter
        self.max_sentence_len = 80
        self.word_size = 50 #词向量维度
        self.position_size = int(15*2) #与命名体相对位置的维度
        self.pos_size = 25 #词性维度
        self.input_size = int(self.word_size)
        self.dnc_hidden = 100
        self.attention_hidden =100
        self.relation_classes = 10
        
        
    def inference(self):
        weights = {
            'attention_h': tf.Variable(tf.random_normal([self.dnc_hidden, 1]), name='attention_h_w'),
            'attention_r': tf.Variable(tf.random_normal([self.dnc_hidden*2, 1]), name='attention_r_w'),
            'attention_out': tf.Variable(tf.random_normal([self.dnc_hidden, self.attention_hidden]), name='attention_out_w'),
            'out': tf.Variable(tf.random_normal([self.attention_hidden, self.relation_classes]), name='out_w')
        }
        biases = {
            'attention': tf.Variable(tf.random_normal([self.max_sentence_len]), name='attention'),
            'out': tf.Variable(tf.random_normal([self.relation_classes]), name='out_b')
        }
        #-------------------------------------- inputs ----------------------------------------------
        sentence_input = tf.placeholder('float', [None, self.max_sentence_len, self.input_size], name='sentence_input')
        sentence_input_T = tf.transpose(sentence_input, [1,0,2])
        sentence_len_input = tf.placeholder('int32', [None], name='sentence_len_input')
        dropout_input = tf.placeholder('float', [2], name='dropout_input')
        # 按entity位置在rnn_out中查询，找到entity对应的rnn_out
        entity_input = tf.placeholder('int32', [None,2], 'entity_input')
        entity_input_onehot = tf.one_hot(entity_input, self.max_sentence_len)
        #-------------------------------------- dnc ----------------------------------------------
        
        dnc_out = dnc_train.run_model(sentence_input_T, self.dnc_hidden, sentence_len_input)
        dnc_out = tf.transpose(dnc_out, [1,0,2])
        # 按entity位置在dnc_out中查询，找到entity对应的rnn_out
        entity_rnn_out = tf.matmul(entity_input_onehot,dnc_out)
        entity_rnn_out = tf.concat([entity_rnn_out[:,0,:],entity_rnn_out[:,1,:]],1)
        #-------------------------------------- attention_layer ----------------------------------------------
        with tf.variable_scope('attention_layer'):
            # attention rate:softmax(tanh(rnn_out*W1 + rnn_last*W2 + b))
#             attention_r = tf.matmul(entity_rnn_out, weights['attention_r'])
#             attention_r = tf.matmul(attention_r, tf.zeros([1,self.max_sentence_len])+1) #复制多份，适应句长
            attention_h = tf.reshape(tf.matmul(tf.reshape(dnc_out,[-1,self.dnc_hidden]), weights['attention_h']), [-1,self.max_sentence_len])
            attention_rate = tf.expand_dims(tf.nn.softmax(tf.nn.tanh(attention_h + biases['attention'])) ,1)
            attention_out = tf.matmul(tf.reshape(tf.matmul(attention_rate, dnc_out), [-1,self.dnc_hidden]), weights['attention_out'])
        #-------------------------------------- output ----------------------------------------------
        with tf.variable_scope('output'):
            out_dropout = tf.nn.dropout(attention_out, dropout_input[1])
#             out_dropout = tf.nn.dropout(dnc_out[:,-1,:], dropout_input[1])
            pred_label = tf.matmul(tf.nn.sigmoid(out_dropout), weights['out']) + biases['out']
        return pred_label, sentence_input, sentence_len_input, entity_input, dropout_input
    
    
    def loss_evaluation(self, pred_label):
        #-------------------------------------- Define loss and evaluation --------------------------------------
        correct_label_input = tf.placeholder('int32', [None, self.relation_classes], name='correct_label_input')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_label, labels=correct_label_input))
        return loss, correct_label_input
    
    
    def train(self, loss, learning_rate=None):
        if learning_rate!=None:
            self.learning_rate = learning_rate
#         optimizer = tf.train.GradientDescentOptimizertrainning_rate=self.learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        return optimizer, init
    
