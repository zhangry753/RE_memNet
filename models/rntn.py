'''
Created on 2017年3月23日

@author: zry
'''
import random
import tensorflow as tf
import sonnet as snt
from tensorflow.contrib import rnn

from models.ntm_cell import Neural_Turing_machine as NtmCore
from ntm_test import config
def cosine_distance(a,b):
  dot = tf.reduce_sum(a*b,-1)
  a_norms = tf.sqrt(tf.reduce_sum(a * a, axis=-1))
  b_norms = tf.sqrt(tf.reduce_sum(b * b, axis=-1))
  similarity = dot / (a_norms*b_norms + 1e-6)
  return similarity


class Rntn:
  def __init__(self):
    # Parameters
    self.learning_rate = 0.001
    self.batch_size = 150
    # Network Parameter
    self.max_sentence_len = 80
    self.word_size = 50 #词向量维度
    self.position_size = int(15*2) #与命名体相对位置的维度
    self.pos_size = 25 #词性维度
    self.input_size = int(self.word_size + self.position_size)
    self.birnn_hidden_size = int(80*2)
    self.atn_hidden_size =100
    self.relation_classes = 10
    
    
  def inference(self):
    #-------------------------------------- inputs ----------------------------------------------
    sentence_input = tf.placeholder('float', [None, self.max_sentence_len, self.input_size], name='sentence_input')
    sentence_len_input = tf.placeholder('int32', [None], name='sentence_len')
    sentence_len = sentence_len_input*2
    dropout_input = tf.placeholder('float', [2], name='dropout_input')
    # 按entity位置在rnn_out中查询，找到entity对应的rnn_out
    entity_input = tf.placeholder('int32', [None,2], 'entity_input')
    entity_input_onehot = tf.one_hot(entity_input, self.max_sentence_len)
    #-------------------------------------- bi-rnn ----------------------------------------------
    with tf.variable_scope('bi-rnn') as scope:
      ntm_cell_config={
        "output_size":int(self.birnn_hidden_size/2),
        "memory_size":int(self.input_size*1.5), 
        "memory_length":int(self.max_sentence_len/4), 
        "controller_hidden":int(self.birnn_hidden_size/2),
      }
#       cell_fw = rnn.DropoutWrapper(NtmCore(**ntm_cell_config), output_keep_prob=dropout_input[0])
#       cell_bw = rnn.DropoutWrapper(NtmCore(**ntm_cell_config), output_keep_prob=dropout_input[0])
#       cell_fw = rnn.DropoutWrapper(rnn.BasicLSTMCell(ntm_cell_config["output_size"]), output_keep_prob=dropout_input[0])
#       cell_bw = rnn.DropoutWrapper(rnn.BasicLSTMCell(ntm_cell_config["output_size"]), output_keep_prob=dropout_input[0])
      cell_fw = rnn.DropoutWrapper(rnn.BasicRNNCell(ntm_cell_config["output_size"]), output_keep_prob=dropout_input[0])
      cell_bw = rnn.DropoutWrapper(rnn.BasicRNNCell(ntm_cell_config["output_size"]), output_keep_prob=dropout_input[0])
      lstm_cell_mix = rnn.BasicLSTMCell(self.birnn_hidden_size)
      bi_rnn_outs,_ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw,
          cell_bw, 
          sentence_input, 
          sequence_length=sentence_len, 
          dtype=tf.float32)
      # 获取内存使用情况(输入输出地址)
      cell_size = ntm_cell_config["output_size"]
      mem_len = ntm_cell_config["memory_length"]
      hidden_fw = bi_rnn_outs[0][:,:,:cell_size]
#       self.addr_in_fw = bi_rnn_outs[0][0,:,cell_size : cell_size+mem_len]
#       self.addr_out_fw = bi_rnn_outs[0][0,:,cell_size+mem_len:]
      hidden_bw = bi_rnn_outs[1][:,:,:cell_size]
#       self.addr_in_bw = bi_rnn_outs[1][0,:,cell_size : cell_size+mem_len]
#       self.addr_out_bw = bi_rnn_outs[1][0,:,cell_size+mem_len:]
      
      rnn_hidden,_ = tf.nn.dynamic_rnn(lstm_cell_mix, tf.concat([hidden_fw, hidden_bw], -1), sequence_length= sentence_len_input, dtype=tf.float32)
#       rnn_hidden = tf.concat([hidden_fw, hidden_bw], -1)
      rnn_hidden_flat = tf.reshape(rnn_hidden, [-1, self.birnn_hidden_size])
      rnn_out = tf.nn.relu(snt.Linear(self.birnn_hidden_size)(rnn_hidden_flat))
      rnn_out = tf.reshape(rnn_out, [-1, self.max_sentence_len, self.birnn_hidden_size])
#       rnn_out = rnn_hidden
      rnn_out_last = rnn_out[:,-1:,:]
    #-------------------------------------- attention_layer ----------------------------------------------
    with tf.variable_scope('attention_layer'):
      # atn rate:softmax(rnn_last*W*rnn_out)
#       rnn_out_last = tf.squeeze(rnn_out_last, 1)
#       atn_left = tf.expand_dims(snt.Linear(self.birnn_hidden_size)(rnn_out_last), 1)
#       atn_rate = tf.nn.softmax(tf.matmul(atn_left, rnn_out, adjoint_b=True))
      # atn rate:softmax([rnn_out,rnn_last]*W + b)
#       atn_r = rnn_out_last + tf.zeros(tf.shape(rnn_out)) #复制多份，适应句长
#       atn_concat_flat = tf.reshape(tf.concat([atn_r, rnn_out], -1), [-1, self.birnn_hidden_size*2])
#       atn_layer1 = tf.nn.relu(snt.Linear(128)(atn_concat_flat))
#       atn_layer2 = tf.nn.relu(snt.Linear(32)(atn_layer1))
#       atn_rate = tf.reshape(tf.nn.softmax(snt.Linear(1)(atn_layer2)), [-1,1,self.max_sentence_len])
      # atn rate:softmax(rnn_out*rnn_Last)
#       atn_mul = tf.matmul(rnn_out_last, rnn_out, adjoint_b=True)
#       atn_rate = tf.nn.softmax(atn_mul)
#       # atn rate: softmax(cosine distance)
      rnn_out_last = rnn_out_last + tf.zeros(tf.shape(rnn_out)) #复制多份，适应句长
      cos_dist = cosine_distance(rnn_out, rnn_out_last)
      atn_rate = tf.nn.softmax(snt.Linear(self.max_sentence_len)(cos_dist))
      atn_rate = tf.expand_dims(atn_rate, 1)
      
      atn_merge = tf.squeeze(tf.matmul(atn_rate, rnn_out), 1)
      atn_out = tf.nn.relu(snt.Linear(self.atn_hidden_size)(atn_merge))
    with tf.variable_scope('output'):
#       out_dropout = tf.nn.dropout(tf.concat([rntn_out,attention_out],-1), dropout_input[1])
      out_dropout = tf.nn.dropout(atn_out, dropout_input[1])
      pred_label = snt.Linear(self.relation_classes)(out_dropout)
    return pred_label, sentence_input, sentence_len_input, entity_input, dropout_input
  
  
  def loss_evaluation(self, pred_label):
    #-------------------------------------- Define loss and evaluation --------------------------------------
    correct_label_input = tf.placeholder('int32', [None, self.relation_classes], name='correct_label_input')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_label, labels=correct_label_input))
    return loss, correct_label_input
  
  
  def train(self, loss, learning_rate=None):
    if learning_rate!=None:
      self.learning_rate = learning_rate
#     optimizer = tf.train.GradientDescentOptimizer(learnitrainte=self.learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    return optimizer, init
  
