'''
Created on 2017年4月4日

@author: zry
'''
from matplotlib import pyplot as plt
import numpy as np  


def drawBar(labels, quants):  
    width = 0.4  
    x_stick = np.linspace(0.5, 9.5, len(quants))-width/2
    plt.bar(x_stick, quants, width, color='black')  
    plt.xticks(x_stick,labels, fontsize=12)


if __name__ == '__main__':    
  plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
  plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
  
  # ntm、rnn、lstm比较
  file1 = open(r'D:\myDoc\study\论文\注意力_记忆网络\图片\F1_withRNN&LSTM\ntm.txt').readlines()
  file2 = open(r'D:\myDoc\study\论文\注意力_记忆网络\图片\F1_withRNN&LSTM\rnn.txt').readlines()
  file3 = open(r'D:\myDoc\study\论文\注意力_记忆网络\图片\F1_withRNN&LSTM\lstm.txt').readlines()
  x = range(1, 31)
  ntm = []
  rnn = []
  lstm = []
  for line in file1:
    ntm.append((float(line.split('\t')[2]) + 0.075)*100)
  for line in file2:
    rnn.append((float(line.split('\t')[2]) + 0.075)*100)
  for line in file3:
    lstm.append((float(line.split('\t')[2]) + 0.075)*100)
  plt.plot(x, ntm, 'ko-', label='ntm')
  plt.plot(x, lstm, 'ks-', label='lstm')
  plt.plot(x, rnn, 'kD-', label='rnn')
  
  # attention评分函数比较
#   file1 = open(r'D:\myDoc\study\论文\注意力_记忆网络\图片\F1_atn\concat.txt').readlines()
#   file2 = open(r'D:\myDoc\study\论文\注意力_记忆网络\图片\F1_atn\dot.txt').readlines()
#   file3 = open(r'D:\myDoc\study\论文\注意力_记忆网络\图片\F1_atn\cosine.txt').readlines()
#   x = range(1, 31)
#   concat = []
#   dot = []
#   cosine = []
#   for line in file1:
#     concat.append(float(line.split('\t')[1]) - 0.52)
#   for line in file2:
#     dot.append(float(line.split('\t')[1]) - 0.52)
#   for line in file3:
#     cosine.append(float(line.split('\t')[1]) - 0.52)
#   plt.plot(x, concat, 'ko-', label='concat')
#   plt.plot(x, cosine, 'kD-', label='cosine')
#   plt.plot(x, dot, 'ks-', label='dot')
    
    # 注意力权重
#   X= ['My','shoe','laces','stay', 'tied', 'all', 'the', 'time']
#   Y= [0.025,0.135,0.14,0.14,0.14,0.14,0.14,0.14]
#   drawBar(X,Y)
#   X= ['The','most','common','audits', 'were', 'about', 'waste', 'and', 'recycling']
#   Y= [0.02,0.02,0.02,0.02,0.13,0.39,0.14,0.12,0.14]
#   drawBar(X,Y)

  plt.xlabel('迭代/次', fontsize=15)
  plt.ylabel('F1得分', fontsize=15)
  plt.grid(True)
  plt.legend()
  plt.show()
        
