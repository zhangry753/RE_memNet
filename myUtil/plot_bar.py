'''
Created on 2017年4月4日

@author: zry
'''
from matplotlib import pyplot as plt
import numpy as np  


def draw(labels,quants,title=None, xlabel=None, ylabel=None):  
    width = 0.4  
    ind = np.linspace(0.5,9.5,len(quants))  
    # make a square figure  
    fig = plt.figure(1)  
    ax  = fig.add_subplot(111)  
    # Bar Plot  
#     ax.plot(ind-width/2,quants,width,color='blue')
    # Line Chart   
    ax.plot(ind,quants,color='blue')  
    # Set the ticks on x-axis  
    ax.set_xticks(ind)  
    ax.set_xticklabels(labels, fontsize=15)
    # lims 
    ax.set_ylim(70,100)  
    # labels  
    if xlabel!=None:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel!=None:  
        ax.set_ylabel(ylabel, fontsize=20)  
    # title
    if title!=None:    
        ax.set_title(title, bbox={'facecolor':'0.8', 'pad':5}, fontsize=23)  
    plt.grid(True)  
    plt.show()  
    plt.close() 



if __name__ == '__main__':    
    file1 = open(r'C:\Users\zry\Desktop\1.txt').readlines()
#     X= ['My','shoe','laces','stay', 'tied', 'all', 'the', 'time']
#     Y= [0.025,0.135,0.14,0.14,0.14,0.14,0.14,0.14]
    x=['2:8','3:7','4:6','5:5','6:4','7:3','8:2']
    y=[85.6,85.8,86.3,86.0,85.5,85.1,84.7]
    draw(x,y,'Proportion of Attention and Tensor','proportion(attention:tensor)','F1-score')
        
