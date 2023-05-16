import csv
from itertools import product
from statannotations.Annotator import Annotator

import matplotlib.pyplot as plt
import matplotlib
import pandas
from matplotlib import rc
#matplotlib.rcParams['text.usetex'] = True
import numpy as np
import os
import glob
import matplotlib.cm as cm
import pandas 
# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import seaborn as sns

index = 0


#videos = sorted(os.listdir('final_dataset'))
videos_1 = ['data_video1', 'data_video2', 'data_video3']
videos_1_names = ['Video1', 'Video2', 'Video3']
#videos_2 = ['anon010','anon012','anon016','anon016_CLIP18' ]
#videos_2_names = ['video5', 'video6', 'video7', 'video8']
list_figures = [videos_1]
list_figures_names = [videos_1_names]
list_tries = ['brisk', 'orb', 'sift', 'SP', 'SG_trained']

#pairs=[
    #(("video1", "brisk"), ("video1", "orb")),
    #(("video1", "SG"), ("video1", "SG_trained")),
    #(("video1", "brisk"), ("video1", "SG_trained")),
    #(("video2", "brisk"), ("video2", "orb")),
    #(("video2", "SG"), ("video2", "SG_trained")),
    #(("video2", "brisk"), ("video2", "SG_trained")),
    #(("video3", "brisk"), ("video3", "orb")),
    #(("video3", "SG"), ("video3", "SG_trained")),
    #(("video3", "brisk"), ("video3", "SG_trained")),
    #]
    
pairs = [("brisk","orb"), ("orb","sift"), ("SG","SG_trained"), ("sift","SG"), ("sift","SG_trained"), ("orb","SG_trained")]

WIDTH = 0.25
colorlist = ['#F9EFAE', '#FADCDA', '#C7AFF9', '#CAEDF7', '#DAF2D9', '#00b4bb', '#00d0a3', '#00e976', '#44ff29']
colorlist = ['#D6E1F1', '#8FD3FB', '#507ADD', '#0E65B8', '#123EE7', '#5593DE', '#81C9E5', '#2878E6', '#2B41E9']
colorlist = ['#D6E1F1', '#96CBFF', '#328EE6', '#0053B4', '#002C80', '#2875CE', '#81C9E5', '#2878E6', '#2B41E9']


p_value_matrix = np.zeros((10, 1))


for index,figure in enumerate(list_figures):

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    # plt.subplots_adjust(left=0.2, right=0.9, top=1.0, bottom=0.6)
    fig.tight_layout()
    color = plt.get_cmap("Set1")
    top = 1.0
    bottom = -0.1
    c = 0

    for i, video_name in enumerate(figure):

        for n,name_try in enumerate(list_tries):
            path_data = os.path.join(os.getcwd(),'metriche', video_name,'ssim_matrix_sp.xlsx')
            if os.path.exists(path_data):
                datax = np.array(pandas.read_excel(path_data))[1:,n+1]
                data = np.squeeze(datax)
                data_tot = np.array(pandas.read_excel(path_data))
                
            else:
                data = np.zeros(10)     
                
            plot = plt.boxplot(data, positions=[i+n*WIDTH+c], widths=0.2, patch_artist=True,
                            boxprops=dict(facecolor=colorlist[n], color='k'), medianprops=dict(color='k'),
                        flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[n]})                 
            

            index = index +1
            
        print(data_tot.shape)
        
        numb = 0
        tmp = 5
        
        for j in range (1, 5):
            for k in range(1, tmp):
              data1 = data_tot[1:,j]
              data2 = data_tot[1:,k+j]
              stat, p = wilcoxon(data1, data2)
              #stat, p = mannwhitneyu(data1, data2)
              print(j,k+j)
              print(p)
              p_value_matrix[numb] = p
              numb = numb +1
            tmp = tmp -1 
          
        #df = pandas.DataFrame(p_value_matrix)
        #path = os.path.join("/home/amdeluca/metriche/" + video_name + "/p_value_matrix.xlsx")
        #df.to_excel(excel_writer=path)
        
        c = c+0.75

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_axisbelow(True)
    #ax.set_ylabel("s")
    ax.set_ylim([bottom, top])

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    #ax.set_xticks([2,6,10,14,18,22,26,30])
    #ax.set_xticks([2,6,10])
    #ax.set_xticks(fontsize=14, rotation=90)
    #ax.set_yticks(ticks=13)
    ax.set_xticks([i+0.5+0.75*i for i in range(0,3,1)])
    list_names = ['Video1', 'Video2', 'Video3']
    ax.set_xticklabels([x for x in list_names], fontsize=13)
    plt.setp(ax.get_xticklabels(), ha="center")
    plt.ylim([0, 1])
    plt.ylabel('Structural Similarity (s)', fontsize=15)
    ax.yaxis.set_ticks_position('none')

            
    
    #fig.legend(loc='upper left', bbox_to_anchor=(1.03, 1))

    fig.savefig('metric_mosaicking_sp_new'+str(index)+'.png', bbox_inches='tight')
    print('done')
    plt.close(fig)




