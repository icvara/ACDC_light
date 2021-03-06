#ploting parameter

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import os
from collections import Counter
import sys
from scipy.signal import argrelextrema
from matplotlib.colors import LogNorm, Normalize
import multiprocessing
import time
from functools import partial


filename="ACDC_X21ind"

n=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
#n=['7']

#sys.path.insert(0, '/users/ibarbier/AC-DC/'+filename)
sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/AC-DC/'+filename)
import model_equation as meq
parlist=meq.parlist


######################################################################33
#########################################################################
###########################################################################

def load(number= n,filename=filename,parlist=parlist):
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    
    path = filename+'/smc/pars_' + number + '.out'
    dist_path = filename+'/smc/distances_' + number + '.out'

    raw_output= np.loadtxt(path)
    dist_output= np.loadtxt(dist_path)
    df = pd.DataFrame(raw_output, columns = namelist)
    df['dist']=dist_output
    df=df.sort_values('dist',ascending=False)
    distlist= sorted(df['dist'])
    p=[]
    for dist in distlist:
        
        p_0=df[df['dist']==dist]
        p0=[]
        for n in namelist:
          p0.append(p_0[n].tolist()[0])
        p0=pars_to_dict(p0,parlist)
        p.append(p0)

    
    return p, df 




def pars_to_dict(pars,parlist):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars
##plotting part


def plot(ARA,p,name,nb,tt=120):
    #ARA=np.logspace(-4.5,-2.,1000,base=10)
    for i,par in enumerate(p):
        

        X,Y,Z = meq.model(ARA,par,totaltime=tt)
        df_X=pd.DataFrame(X,columns=ARA)
        df_Y=pd.DataFrame(Y,columns=ARA)
        df_Z=pd.DataFrame(Z,columns=ARA)


        plt.subplot(len(p),3,(1+i*3))
        sns.heatmap(df_X, cmap="Reds", norm=LogNorm())
        plt.xticks([])
        plt.ylabel('time')
        plt.subplot(len(p),3,(2+i*3))
        sns.heatmap(df_Y, cmap ='Blues', norm=LogNorm())
        plt.xticks([])
        plt.yticks([])
        plt.subplot(len(p),3,(3+i*3))
        sns.heatmap(df_Z, cmap ='Greens', norm=LogNorm())
        plt.xticks([])
        plt.yticks([])



    #plt.savefig(name+"/plot/"+nb+'_heatmap'+'.pdf', bbox_inches='tight')
    plt.savefig(name+"/plot/heatmap/"+nb+'_heatmap'+'.png', bbox_inches='tight')
    #plt.show()
    plt.close()




def par_plot(df,name,nb,parlist,namelist):
    #plt.plot(df['K_ARAX'],df['K_ARAY'],'ro')
    fonts=5
 
    for i,par1 in enumerate(namelist):
        for j,par2 in enumerate(namelist):
            plt.subplot(len(namelist),len(namelist), i+j*len(namelist)+1)
            if i == j :
                plt.hist(df[par1])
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
            else:
                plt.scatter(df[par1],df[par2], c=df['dist'], s=0.1, cmap='viridis')# vmin=mindist, vmax=maxdist)
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
                plt.ylim((parlist[j]['lower_limit'],parlist[j]['upper_limit']))
            if i > 0 and j < len(namelist)-1 :
                plt.xticks([])
                plt.yticks([])
            else:
                if i==0 and j!=len(namelist)-1:
                    plt.xticks([])
                    plt.ylabel(par2,fontsize=fonts)
                    plt.yticks(fontsize=fonts,rotation=90)
                if j==len(namelist)-1 and i != 0:
                    plt.yticks([])
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                if i==0 and j==len(namelist)-1:
                    plt.ylabel(par2,fontsize=fonts)
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                    plt.yticks(fontsize=fonts,rotation=90)                 
  #  plt.savefig(name+"/plot/"+nb+'_par_plot.pdf', bbox_inches='tight')
    plt.savefig(name+"/plot/par/"+nb+'_par_plot.png', bbox_inches='tight', dpi=300)

    plt.close()
    #plt.show()
    







##############################################################################################################3   

if __name__ == "__main__":
   
    if os.path.isdir(filename+'/plot') is False: ## if 'smc' folder does not exist:
        os.mkdir(filename+'/plot') ## create it, the output will go there
    
    ARA=meq.ARA
    ARA=np.logspace(-8,-2.,200,base=10)

    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    

    n=['final']
    p, pdf= load(n[0],filename,meq.parlist)
    df=pdf

    plot(ARA,[p[0],p[250],p[500],p[750],p[999]],filename,"moretime",tt=500)

    '''
   # ARA=ARA[[0,4,5,9]]
    for i in n:
      p, pdf= load(i,filename,meq.parlist)
    
      plot(ARA,[p[0],p[250],p[500],p[750],p[999]],filename,i)
      par_plot(pdf,filename,i,meq.parlist,namelist)

    '''
 

    
    