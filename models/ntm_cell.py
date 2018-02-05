'''
Created on 2017年10月31日

@author: zry
'''
import collections
import sonnet as snt
import tensorflow as tf

NTM_State = collections.namedtuple('NTM_State', ('memory', 'memory_output', 'controller_state'))

class Neural_Turing_machine(snt.RNNCore):
  
  def __init__(self,
               output_size, 
               memory_size, 
               memory_length, 
               controller_hidden,
               name="NTM"):
    super(Neural_Turing_machine, self).__init__(name=name)
    self._controller_hidden = controller_hidden
    controller_hidden += 2*memory_length
    self._controller = snt.LSTM(controller_hidden)
    self._output_size = output_size
    self._mem_size = memory_size
    self._mem_len = memory_length
        
    
  def _build(self, inputs, prev_state, scope=None):
    prev_memory = prev_state.memory
    prev_memory_output = prev_state.memory_output
    prev_controller_state = prev_state.controller_state
    
    controller_input = tf.concat([inputs, prev_memory_output], -1)
    controller_output, controller_state = self._controller(controller_input, prev_controller_state)
    ctr_hidden_out = controller_output[:, :self._controller_hidden]
    ctr_hidden_addr_in = controller_output[:, self._controller_hidden : self._controller_hidden+self._mem_len]
    ctr_hidden_addr_out = controller_output[:, self._controller_hidden+self._mem_len :]
    # 存入内存
    mem_addr_in = tf.nn.softmax(snt.Linear(self._mem_len)(ctr_hidden_addr_in))
    mem_addr_in_3d = tf.expand_dims(mem_addr_in, 2)
    new_mem_value = tf.nn.relu(snt.Linear(self._mem_size)(inputs))
    new_mem_value = tf.expand_dims(new_mem_value, 1)
    new_memory = mem_addr_in_3d*new_mem_value + (1-mem_addr_in_3d)*prev_memory
    # 从内存中取出
    mem_addr_out = tf.nn.softmax(snt.Linear(self._mem_len)(ctr_hidden_addr_out))
    mem_addr_out_3d = tf.expand_dims(mem_addr_out, 1)
    mem_output = tf.reshape(tf.matmul(mem_addr_out_3d, prev_memory), [-1,self._mem_size])
    # 输出
    output = tf.concat([ctr_hidden_out, mem_output], 1)
    output = snt.Linear(self._output_size, name='output_linear')(output)
    
    return tf.concat([output, mem_addr_in, mem_addr_out],-1), NTM_State(
        memory=new_memory,
        memory_output=mem_output,
        controller_state=controller_state
        )

  @property
  def state_size(self):
    return NTM_State(
        memory=tf.TensorShape([self._mem_len,self._mem_size]),
        memory_output=tf.TensorShape([self._mem_size]),
        controller_state=self._controller.state_size
        )

  @property
  def output_size(self):
    return tf.TensorShape([self._output_size+\
                           self._mem_len+\
                           self._mem_len
                           ])
  
  
