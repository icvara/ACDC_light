'''

Do call the difference analysis of ACDC


'''
import random
import numpy as np
import matplotlib.pyplot as plt
from plot import *

import seaborn as sns

import statistics


from scipy.stats import gaussian_kde
from matplotlib import colors


color=sns.color_palette("colorblind")


'''
================================================================================

SET UP 

===============================================================================
'''
inch_cm =2.54


filename="ACDC_X_1ind_new"
n=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','final']

#filename="ACDC_X2"
#n=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
#n=['7']


#sys.path.insert(0, '/users/ibarbier/AC-DC/'+filename)
sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/ACDC_light/'+filename)
import model_equation as meq
parlist=meq.parlist



'''
================================================================================

Analyse 0 : distance

===============================================================================
'''

'''
n=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','final']

coll = sns.color_palette("flare", len(n))
c=0
for i in n:
    p, df= load(i,filename,meq.parlist)
    #sns.set_style('whitegrid')
    sns.kdeplot( data=df['dist'], color=coll[c])
    #sns.distplot(np.log(df['dist']), color=coll[c], hist=False)    
    #plt.hist(df['dist'], color=coll[c],bins=10, density=True, alpha=0.5)
    med=np.median(df['dist'])
    plt.axvline(med, linestyle='--', label=i, lw=.8, color=coll[c])
    c+=1
plt.legend()
plt.axvline(1, color='k', linestyle='solid', label='', lw=.8)
plt.savefig(filename+"/_distance.pdf", dpi=300)
plt.show()
stop
'''


'''
================================================================================

Analyse 1 : heatmap

===============================================================================
'''

'''
n=['1','5','10','final']
heatmap_allroound(meq.ARA,filename,meq.parlist,n,meq)
plt.savefig(filename+"_heatmap.pdf", dpi=300)
plt.show()
'''



'''
================================================================================

Analyse 2 : par Space

===============================================================================
'''

#PARPLOT
'''
n=['final']
p, df= load(n[0],filename,meq.parlist)
par_plot(df,parlist)
plt.savefig(filename+"_parspace.pdf", dpi=300)
plt.show()
'''



'''
================================================================================

Analyse 3 : par stats

still need to do

===============================================================================
'''

'''
n='final'
p, df= load(n,filename,meq.parlist)

for pm in parlist:
    nn=pm['name']
    print(nn, np.round(np.mean(df[nn]),2), np.round(np.std(df[nn]),2), np.round(np.median(df[nn]),2) )
barplot(df,parlist)
plt.savefig(filename+"_barplot.pdf", dpi=300)
plt.show()
'''









'''
================================================================================================================================================

Analyse 4 : 1 vs 2 ind comparisons

===============================================================================================================================================
'''




'''
filename1="ACDC_X_1ind_new"
filename2="ACDC_X_2ind_new"


#filename="ACDC_X2"
#n=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
#n=['7']

#sys.path.insert(0, '/users/ibarbier/AC-DC/'+filename)
sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/ACDC_light/'+filename1)
import model_equation as meq1
parlist1=meq1.parlist

parlist2 = [ 
    #first node X param
    {'name' : 'K_a', 'lower_limit':-10.0,'upper_limit':-1.0}, 
    {'name' : 'n_a','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XY','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_XY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XZ','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_XZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'B_X','lower_limit':0.0,'upper_limit':4.0},

    #Seconde node Y param
    {'name' : 'K_b', 'lower_limit':-4.0,'upper_limit':-1.0}, 
    {'name' : 'n_b','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_YZ','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_YZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'B_Y','lower_limit':0.0,'upper_limit':4.0},

    #third node Z param
    {'name' : 'K_ZX','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_ZX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'B_Z','lower_limit':0.0,'upper_limit':4.0},
]


parlistC = [ 
    #first node X param
    #{'name' : 'K_a', 'lower_limit':-10.0,'upper_limit':-1.0}, 
    {'name' : 'n_a','lower_limit':0.5,'upper_limit':2.0},
    #{'name' : 'K_XY','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_XY','lower_limit':0.5,'upper_limit':2.0},
    #{'name' : 'K_XZ','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_XZ','lower_limit':0.5,'upper_limit':2.0},
    #{'name' : 'B_X','lower_limit':0.0,'upper_limit':4.0},

    #Seconde node Y param
    #{'name' : 'K_b', 'lower_limit':-4.0,'upper_limit':-1.0}, 
    #{'name' : 'n_b','lower_limit':0.5,'upper_limit':2.0},
    #{'name' : 'K_YZ','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_YZ','lower_limit':0.5,'upper_limit':2.0},
    #{'name' : 'B_Y','lower_limit':0.0,'upper_limit':4.0},

    #third node Z param
    #{'name' : 'K_ZX','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_ZX','lower_limit':0.5,'upper_limit':2.0},
    #{'name' : 'B_Z','lower_limit':0.0,'upper_limit':4.0},
]

#parlistC = parlist1
mm="n"

n='final'
par1, df1= load(n,filename1,parlist1)
par2, df2= load(n,filename2,parlist2)


cl=None

fig, axs = plt.subplots(len(parlistC),len(parlistC),figsize=(len(parlistC),len(parlistC)))#, constrained_layout=True, figsize=(10/inch_cm,10/inch_cm))

for x in np.arange(len(parlistC)):
    for y in np.arange(len(parlistC)):

        px=parlistC[x]['name']
        py=parlistC[y]['name']

        if x==y:
            #axs[x,y] = sns.distplot(df_par[px])
            axs[x,y].hist(df1[px],bins=20, density=True, alpha=0.5,color='r')
            axs[x,y].hist(df2[px],bins=20, density=True, alpha=0.5,color='b')

            axs[x,y].set_xlim(parlistC[x]['lower_limit'],parlistC[x]['upper_limit'])
            #sns.kdeplot(ax=axs[x, y],x=df_par[px])

        elif y>x:
            axs[x,y].axis('off')
            #axs[x,y].hist2d(df2[py],df2[px],bins=15, norm = colors.LogNorm(), alpha=1 , cmap="Blues")




        elif y<x:
            #z = gaussian_kde(df_par[px])(df_par[py])
            #idx = z.argsort()
            #xx, yy, z = df_par[px][idx], df_par[py][idx], z[idx]
            #axs[x,y].scatter(xx,yy,c=z, s=0.1, cmap='viridis')# vmin=mindist, vmax=maxdist)
            if cl==None:
                axs[x,y].hist2d(df2[py],df2[px],bins=15, norm = colors.LogNorm(), alpha=1 , cmap="Blues")
                axs[x,y].hist2d(df1[py],df1[px],bins=15, norm = colors.LogNorm(), alpha=1 , cmap="Reds")

            else:
                axs[x,y].scatter(df1[py],df2[px],c='r', s=0.1, alpha=0.1)# vmin=mindist, vmax=maxdist)
                axs[x,y].scatter(df2[py],df2[px],c='b', s=0.1, alpha=0.1)# vmin=mindist, vmax=maxdist)


            axs[x,y].set_xlim(parlistC[y]['lower_limit'],parlistC[y]['upper_limit'])
            axs[x,y].set_ylim(parlistC[x]['lower_limit'],parlistC[x]['upper_limit'])


        
        niceaxis(axs,x,y,px,py,parlistC,parlistC,12)

plt.subplots_adjust(wspace=None, hspace=None)
plt.savefig("_1vs2_"+mm+"_parspace.pdf", dpi=300)
#plt.savefig("_1vs2_parspace.png", dpi=300)

plt.show()
'''

####################################################
###############################compare K
######################################################3

'''
fig, axs = plt.subplots(1,2)#,figsize=(len(parlistC),len(parlistC)))#, constrained_layout=True, figsize=(10/inch_cm,10/inch_cm))


for x in np.arange(len(parlistC)):

        px=parlistC[x]['name']
        axs[0].hist(df1[px],bins=50, density=True, alpha=0.5 ,label=px)
        axs[1].hist(df2[px],bins=50, density=True, alpha=0.5, label=px)

axs[0].set_xlim(parlistC[x]['lower_limit'],parlistC[x]['upper_limit'])
axs[1].set_xlim(parlistC[x]['lower_limit'],parlistC[x]['upper_limit'])
axs[0].legend()
axs[0].set_title("one inducer")
axs[1].set_title("two inducers")
axs[0].set_ylabel("density")



plt.savefig("_1vs2_"+mm+"_pars.pdf", dpi=300)

plt.show()

stop
'''


'''
================================================================================

Analyse 5 : Bifurcation

dont know this part

===============================================================================
'''

'''
n='final'
par, df= load(n,filename,meq.parlist)
ARA=meq.ARA
ARA=np.logspace(-7,-1.,10,base=100)
i=0
bifuplot_all(ARA,filename,par[i],meq,0,i)
bifuplot_all(ARA,filename,par[i],meq,1,i)
bifuplot_all(ARA,filename,par[i],meq,2,i)
plt.show()
'''

'''
##########################################single plot
n='final'
par, df= load(n,filename,meq.parlist)
ARA=meq.ARA
ARA=np.logspace(-5,-1.,10,base=10)
i=0
bifuplot(ARA,filename,par[i],meq,i)
plt.show()
'''
'''
############################################grid
n='final'
par, df= load(n,filename,meq.parlist)
pp=np.arange((25))
s=int(np.sqrt(len(pp)))
fig,axs= plt.subplots(s,s, figsize=(s,s))
ARA=meq.ARA
x=0
y=0
for i,ni in enumerate(pp):
    bifuplot_grid(axs,x,y,ARA,filename,p[ni],meq,ni)
    axs[x,y].text(ARA[-2],10e-5, ("p "+str(ni)),fontsize=8,ha='center', va='center')
    niceaxis(axs,x,y,ni,ni,pp,pp,6)
    x+=1
    if x==s:
        x=0
        y+=1

plt.show()

'''



'''
============================================================================================================================================================================================================

chek ACDC behavior

============================================================================================================================================================================================================
'''


'''
ACDC behavior is  stable state, hopf, oscil , homoclinic or hopf, stable state in aRA gardient

1: stable
2: oscil
3: unsatble

4:stable+stable
8:oscil + stable
20:tristability

#transition
stable(1) to bistable(4) , saddle : 3
stable(1) to osc+stable (8) : homoclinic :7    #will probably only happens is resolution is low
osc (2) to osc+stavle(8) : saddle/homoclinic/ruben :  6 

osc(2) to bistable(4) : hopf and saddle : 2  #possible went to homclinic before
stable(1) to oscilation(2) : hopf :1

osc+stable(8) to bistable (4) :  hopf?:  4  #don't know if exist


tri(20) to stable (1) :  saddle?:  19 
tri(20) to oscil (2) :  saddle?:  18 
tri(20) to bistable (4) :  saddle?:  16 
tri(20) to oscil + stable (8) :  saddle?:  12 


'''


n='final'
par, df= load(n,filename,meq.parlist)
ARA=meq.ARA
ARA=np.logspace(-7,-1,100,base=10)
mid=40



#LAUNCH THE CHARACTERIZTION...take >10h
#############################################################3
#getBehavior(pselected,ARA,filename,meq,"all")





#Analyzed
#############################################################3

state,bifu,bist = loadBifurcation(filename,"all")

#========================
#
#1. look at multistability
#
#======================

#AC + DC
############

number=[]
#entering
to=np.apply_along_axis(countbis, 1, bist[:,:mid]) #before
idx_in=np.where(to>0)[0]
#leaving
to=np.apply_along_axis(countbis, 1, bist[:,mid:]) #before
idx_out=np.where(to>0)[0]
#both
idx_both=getidentic(idx_in,idx_out)
#all
to=np.apply_along_axis(countbis, 1, bist[:,:]) #before
idx_all=np.where(to>0)[0]

idx_out=getidiff(idx_both,idx_out)
idx_in=getidiff(idx_both,idx_in)

number.append(len(idx_in))
number.append(len(idx_out))
number.append(len(idx_both))
number.append(len(idx_all))
print("bistability: " + str(number))


#2 no AC+DC
#######################################################################


#2.2 fake
tf=np.apply_along_axis(countfake, 1, state) #before
idx_fake=np.where(tf==1)[0]
#idx_fake=getidiff(idx_tris,idx_fake)
print("fake: " + str(len(idx_fake)))

#update no bis index
idx_nobis=getidiff(idx_all,np.arange(5000))
idx_nobis=getidiff(idx_fake,idx_nobis)
#idx_nobis=getidiff(idx_tris,idx_nobis)

#2.3 bistability outside oscillation
to=np.apply_along_axis(countsaddle2, 1, bifu[:,:mid]) #
idx_sd_in=np.where(to>0)[0]
to=np.apply_along_axis(countsaddle2, 1, bifu[:,mid:]) #
idx_sd_out=np.where(to>0)[0]
to=np.apply_along_axis(countsaddle2, 1, bifu[:,:]) #
idx_sd_all=np.where(to>0)[0]
idx_sd_all=getidentic(idx_sd_all,idx_nobis)
idx_sd_in=getidentic(idx_sd_in,idx_nobis)
idx_sd_out=getidentic(idx_sd_out,idx_nobis)
idx_sd_both=getidentic(idx_sd_in,idx_sd_out)
number=[]
idx_sd_out=getidiff(idx_sd_both,idx_sd_out)
idx_sd_in=getidiff(idx_sd_both,idx_sd_in)
number.append(len(idx_sd_in))
number.append(len(idx_sd_out))
number.append(len(idx_sd_both))
number.append(len(idx_sd_all))
print("no bis saddle outside: " + str(number))


#update
idx_nobis=getidiff(idx_sd_all,idx_nobis)

#2.4 bistability int homoclinice
to=np.apply_along_axis(countsaddle, 1, bifu[:,:]) #
idx_hc_all=np.where(to>0)[0]
to=np.apply_along_axis(countsaddle, 1, bifu[:,:mid]) #
idx_hc_in=np.where(to>0)[0]
to=np.apply_along_axis(countsaddle, 1, bifu[:,mid:]) #
idx_hc_out=np.where(to>0)[0]
idx_hc_in=getidentic(idx_hc_in,idx_nobis)
idx_hc_out=getidentic(idx_hc_out,idx_nobis)
idx_hc_both=getidentic(idx_hc_in,idx_hc_out)
idx_hc_all=getidentic(idx_hc_all,idx_nobis)
number=[]
idx_hc_out=getidiff(idx_hc_both,idx_hc_out)
idx_hc_in=getidiff(idx_hc_both,idx_hc_in)
number.append(len(idx_hc_out))
number.append(len(idx_hc_in))
number.append(len(idx_hc_both))
number.append(len(idx_hc_all))
print("no bis Hc: " + str(number))

#2.5 leftover
idx_nobis=getidiff(idx_hc_all,idx_nobis)
idx_hpf_all=np.array(list(set(np.where(state[:,:]==2)[0])))
idx_hpf_all=getidentic(idx_hpf_all,idx_nobis)
idx_nobis=getidiff(idx_hpf_all,idx_nobis)
print("no bis, hopf: "+ str(len(idx_hpf_all)))
print("other: " + str(len(idx_nobis)))


#2.1 tristability?
tt=np.apply_along_axis(counttris, 1, state[:,:]) #before
idx_tris=np.where(tt>0)[0]
idx_tris1=np.where(tt>0)[0]

#idx_tris=getidiff(idx_all,idx_tris)
idx_tris1=getidiff(idx_all,idx_tris1)
idx_tris=getidentic(idx_all,idx_tris)
print("tristability: " + str(len(idx_tris)))

#HERE NEED TO CHECK

###################################################################################
#plotting time
################################################3

y = np.array([len(idx_all),#len(idx_out),len(idx_both),
    len(idx_sd_all),
    len(idx_hc_all),len(idx_hpf_all), 
    np.sum((len(idx_nobis), len(idx_fake)))])#, len(idx_tris)])
print(np.sum(y))




mylabels = ["multistablity - AC-DC",#,"multistablity - AC-DC (h)","multistablity - AC-DC (b)",
"multistablity - DC" , " multistablity - homoclinic", "no multistability - hopf",
 "no AC" ]#, "tristability need to do" ]

mycolors = [color[0],# color[0], color[0], 
color[2], color[2], color[3], 
color[7],color[8]]

wp = { 'linewidth' : 1, 'edgecolor' : "black" }
plt.pie(y, labels = mylabels, colors = mycolors,wedgeprops = wp, startangle=180, counterclock=False)
    #autopct="%1.1f%%", pctdistance=1.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(filename+"/plot/" + "_stability_piechart.pdf", dpi=300)
plt.show() 

'''
for ii,i in enumerate([idx_all,idx_sd_all,idx_hc_all,idx_hpf_all,idx_nobis,idx_fake,idx_tris]):
    if len(i)>0:
        pi= random.choices(i, k=25)
        plot_bifu_grid(pi,par,ARA,filename,meq)
        plt.savefig(filename+"/plot/bifu/" +str(ii)+ "_bifuplot_" + ".pdf", dpi=300)
        plt.close()
    #plt.show()
'''

##############################################################################33
####################################################################################


#========================
#
#2. entering AC!!!
#
#======================

print("--------------------------------------")

hopf=np.apply_along_axis(counthopf, 1, bifu[:,:]) #before
bist=np.apply_along_axis(countbis, 1, bifu[:,:]) #before
saddle=np.apply_along_axis(countsaddleoscil, 1, bifu[:,:]) #before

#some imprecision in hopf... can double check but max is 2

idx0=np.where(hopf==0)[0]
idx1=np.where(hopf==1)[0]
idx2=np.where(hopf>1)[0]

idx3=np.where(saddle==0)[0]
idx4=np.where(saddle==1)[0]
idx5=np.where(saddle>1)[0]



y=[]
l=[]

for ii,i in enumerate([idx0,idx1,idx2]):
    for jj,j in enumerate([idx3,idx4,idx5]):

        idx=getidentic(i,j)
        idx=getidentic(idx,idx_all)
        print(ii,jj,len(idx))
        '''
        if len(idx)>0:
            pi= random.choices(idx, k=25)
            plot_bifu_grid(pi,par,ARA,filename,meq)
            plt.savefig(filename+"/plot/bifu/" +str([ii,jj])+ "_enterAC.pdf", dpi=300)
            plt.close()

            #plt.show()
        '''

        y.append(len(idx))
        text="hopf: "+ str(ii) + "\nSaddle: " + str(jj)
        l.append(text)

newsort=[0,1,2,3,6,4,5,7,8]
mylabels=[]
ysort=[]
for i in newsort:
    mylabels.append(l[i])
    ysort.append(y[i])


mycolors = [color[7],# color[0], color[0], 
color[2], color[2], color[3], color[3], color[4] , color[4],
color[8],color[8]]

wp = { 'linewidth' : 1, 'edgecolor' : "black" }
plt.pie(ysort, labels = mylabels, colors = mycolors,wedgeprops = wp, startangle=180, counterclock=False)
    #autopct="%1.1f%%", pctdistance=1.5)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(filename+"/plot/" + "_bifu_piechart.pdf", dpi=300)
plt.show() 


'''
idx=getidentic(idx0,idx3)
idx=getidentic(idx,idx_all)
pi=idx


print(pi)


plot_bifu_grid(pi,par,ARA,filename,meq)
plt.savefig(filename+"/plot/bifu/" +str([0,0])+ "_enterAC.pdf", dpi=300)
plt.close()

kl=2980
#pselected=[par[kl]]
#print(getBehavior(pselected,ARA,filename,meq,"i",False))

print(state[kl])
print(bifu[kl])
print(bist[kl])
'''
stop



'''
================================================================================================================================================

Analyse 6 : Dynmamic behaviour

still need to do

===============================================================================================================================================
'''



n='final'
par, df= load(n,filename,meq.parlist)
state,bifu,bist = loadBifurcation(filename,"all")

ARA=meq.ARA
size=100

#size=100
tt=400#400
coll = sns.color_palette("viridis", size)
ARA=np.logspace(-7,-1.,size,base=10)
#ARA=np.logspace(-7,-2.,size,base=10)

ARA2=np.logspace(np.log10(ARA[2]),np.log10(ARA[3]),10,base=10)

ARA3=np.logspace(np.log10(ARA2[4]),np.log10(ARA2[5]),10,base=10)

#ARA=np.append(ARA2,ARA)
#ARA=np.append(ARA,ARA3)


#size=len(ARA)

idx=3836#4297#301
transient=int(250/meq.dtt)
#bifuplot(ARA,filename,par[i],meq,i)
#plt.show()

pi=par[idx]

period=[]
periodh=[]

ss=meq.findss2(ARA,pi) 


mid=int(size/2)
#mid=40#int(size/2)

ai=6#80
ai=80

initlow= ss[ai,1]+10e-10
print(ss[ai])

X,Y,Z = meq.model(ARA,pi,tt,meq.dtt, init=initlow)

fig,axs= plt.subplots(2,3, constrained_layout=True, figsize=(18,12))

bifuplot_grid(axs,0,0,ARA,filename,pi,meq,dummy=pi)
#axs[0,1].axvline(transient*meq.dtt, c='k')

axs[0,1].imshow(np.log10(X[transient:,:]), aspect="auto", cmap="Reds")
#axs[1,1].imshow(np.log10(Xh[transient:,:]), aspect="auto", cmap="Reds")

for i in np.arange(size):
    axs[0,2].plot(np.arange(X.shape[0])*meq.dtt,X[:,i], c=coll[i], alpha=0.5)
  #  axs[1,2].plot(np.arange(Xh.shape[0])*meq.dtt,X[:,i], c=coll[i], alpha=0.5)
    max_list=argrelextrema(X[transient:,i], np.greater)
    min_list=argrelextrema(X[transient:,i], np.less)
    maxValues=X[transient:,i][max_list]
    minValues=X[transient:,i][min_list]

    if len(max_list[0])>3 and np.abs(maxValues[-2]-minValues[-2])>0.1:
        period.append((max_list[0][-2] - max_list[0][-3])*meq.dtt)
        print(i , (max_list[0][-2] - max_list[0][-3])*meq.dtt)
        #print(np.abs(maxValues[-2]-minValues[-2]))

    else:
        #print(i)
        period.append(0)





axs[1,2].plot(np.arange(transient,X.shape[0])*meq.dtt,X[transient:,41], alpha=0.5)
axs[1,2].plot(np.arange(transient,X.shape[0])*meq.dtt,X[transient:,62], alpha=0.5)
axs[1,2].plot(np.arange(transient,X.shape[0])*meq.dtt,X[transient:,72], alpha=0.5)

#axs[0,0].plot(ARA,l)
#axs[0,0].plot(ARA,h)



axs[1,0].plot(ARA,period,'-k',)#c=coll[i])
axs[1,0].set_ylim(5,30)

#axs[1,0].plot(ARA,periodh,'--k')#c=coll[i])

axs[1,0].set_xscale("log")#c=coll[i])

'''
for i in np.arange(size):
  #  axs[0].plot(X[:,i], c=coll[i])
    #axs[0,1].plot(np.arange(Xh.shape[0])*meq.dtt,Xh[:,i], c=coll[i], alpha=0.5)
    #axs[0,1].imshow(np.arange(Xh.shape[0])*meq.dtt,Xh[:,i], cmap="Reds")#, alpha=0.5)

    #axs[0,2].plot(np.arange(X.shape[0])*meq.dtt,X[:,i], c=coll[i], alpha=0.5)


    axs[1,2].plot(Y[transient:,i],X[transient:,i], c=coll[i], alpha=0.5 )

    axs[1,1].plot(Yh[transient:,i],Xh[transient:,i], c=coll[i], alpha=0.5 )


    max_list=argrelextrema(X[transient:,i], np.greater)
    maxValues=X[transient:,i][max_list]
    period.append( np.nanmax((0,np.mean(max_list[0][1:-1] - max_list[0][:-2])))*meq.dtt)

    max_list=argrelextrema(Xh[transient:,i], np.greater)
    maxValues=Xh[transient:,i][max_list]
    periodh.append( np.nanmax((0,np.mean(max_list[0][1:-1] - max_list[0][:-2])))*meq.dtt)

axs[1,0].plot(ARA,period,'-')#c=coll[i])
axs[1,0].plot(ARA,periodh,'--')#c=coll[i])

axs[1,0].set_xscale("log")#c=coll[i])
'''


axs[0,0].set_xlabel("ARA")
axs[0,0].set_ylabel("X")

axs[1,0].set_xlabel("ARA")
axs[1,0].set_ylabel("period")


axs[0,1].set_xlabel("S")
axs[0,1].set_ylabel("time")


axs[1,1].set_xlabel("Y")
axs[1,1].set_ylabel("X")

axs[1,2].set_xlabel("Y")
axs[1,2].set_ylabel("X")
axs[1,2].set_yscale("log")
#axs[1,2].set_xscale("log")



#axs[0,1].set_yscale("log")
axs[0,2].set_yscale("log")


#axs[1,1].set_yscale("log")
#axs[1,1].set_xscale("log")



#plt.yscale("log")
plt.savefig(filename+"/" + str(idx) + "_detailed.pdf", dpi=300)

plt.show()

ax = plt.axes(projection='3d')
i=4
# Data for a three-dimensional line
for i in np.arange(size):

    z = np.log(Z[transient:,i])
    x = np.log(X[transient:,i])
    y = np.log(Y[transient:,i])
    ax.plot3D(x, y, z, c=coll[i], alpha=0.5)

# Data for three-dimensional scattered points

   # ax.scatter3D(x, y, z, c=coll[i], cmap='Greens',s=0.1)
plt.show()