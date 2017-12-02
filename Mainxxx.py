'''
Created on 2017年10月28日

@author: zry
'''
import tensorflow as tf
import sys
import numpy as np

from models.ntm_cell import Neural_Turing_machine as NTM
import config


def run_model(input_sequence, output_size, sequence_length=None):
  """Runs model on input sequence."""
  model_config = {
      "output_size":output_size,
      "memory_size":config.register_size,
      "memory_length":config.register_num,
      "controller_hidden":config.register_size,
  }
  ntm_core = NTM(**model_config)
  rnn_outputs, _ = tf.nn.dynamic_rnn(
      cell=ntm_core,
      inputs=input_sequence,
      sequence_length=sequence_length,
      time_major=True,
      dtype=tf.float32)
  hidden = rnn_outputs[:,:,:output_size]
  mem_addr_in = rnn_outputs[:,:,output_size : output_size+config.register_num]
  mem_addr_out = rnn_outputs[:,:,output_size+config.register_num:]
  return hidden, ntm_core, mem_addr_in, mem_addr_out


def train(num_training_iterations, report_interval):
  obs_pattern = tf.cast(
          tf.random_uniform(
              [config.max_length,config.batch_size,config.num_bits], minval=0, maxval=2, dtype=tf.int32),
          tf.float32)
  obs = tf.concat([obs_pattern, tf.zeros([config.max_length+5,config.batch_size,config.num_bits])], 0)
  target = tf.concat([tf.zeros([config.max_length,config.batch_size,config.num_bits]), 
                      obs_pattern, 
                      tf.zeros([5,config.batch_size,config.num_bits])], 0)
  output_logits, ntm_core, addr_in, addr_out = run_model(obs, config.num_bits)
  output = tf.nn.sigmoid(output_logits)
  # train loss
#   train_loss = tf.reduce_mean(tf.square(target-output_logits),-1, keep_dims=True)
  train_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output_logits))
  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(train_loss, trainable_variables), config.max_grad_norm)
  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
  optimizer = tf.train.RMSPropOptimizer(config.learning_rate, epsilon=config.optimizer_epsilon)
#   optimizer = tf.train.AdamOptimizer()
  train_step = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step)
  hooks = []
  # scalar
#   tf.summary.scalar('loss', train_loss)
#   hooks.append(
#       tf.train.SummarySaverHook(
#           save_steps=5,
#           output_dir=config.checkpoint_dir+"/logs",
#           summary_op=tf.summary.merge_all()))
  # saver
  saver = tf.train.Saver(max_to_keep=50)
  if config.checkpoint_interval > 0:
    hooks.append(
        tf.train.CheckpointSaverHook(
            checkpoint_dir=config.checkpoint_dir,
            save_steps=config.checkpoint_interval,
            saver=saver))
  # Train.
  with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=config.checkpoint_dir) as sess:
    start_iteration = sess.run(global_step)
    total_loss = 0
    total_dqn_loss = 0
    
    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss = sess.run([train_step, train_loss])
      total_loss += loss
      
      if (train_iteration + 1) % report_interval == 0:
        obs_np, output_np, \
            addr_in_np, addr_out_np =\
            sess.run([obs_pattern, output,
                   addr_in, addr_out])
        print("%d: Avg training loss %f.\t%f.\n "%(
                        train_iteration, total_loss/report_interval, total_dqn_loss/report_interval))
        print(obs_np[:,0,:], output_np[config.max_length:config.max_length*2,0,:], sep='\n')
        print(addr_in_np[:,0,:], addr_out_np[:,0,:], sep='\n')
        sys.stdout.flush()
        total_loss = 0
        total_dqn_loss = 0


def main(unused_argv):
  train(config.num_training_iterations, config.report_interval)


if __name__ == "__main__":
#   sys.stdout = open(r'message.log','w')
  tf.app.run()
  



