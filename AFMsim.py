# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:12:32 2017

@author: pyh
"""
class Gra_fit(object):
    def __init__(self,X,Y,a,b):
        self.input=X
        self.obtain=Y
        self.alpha=a
        self.beta=b
        
    def optimizer(self,learning_rate_alpha=0.01,learning_rate_beta=0.01):
        delta_a=0
        delta_b=0
        for i in range(len(self.input)):            
            delta_a=delta_a+(self.obtain[i]-self.alpha*np.power(self.input[i][0],self.beta)*
                              self.input[i][1])*np.power(self.input[i][0],self.beta)*self.input[i][1]
            delta_b=delta_b+(self.obtain[i]-self.alpha*np.power(self.input[i][0],self.beta)*
                              self.input[i][1])*np.log(self.input[i][0])*self.alpha*np.power(self.input[i][0],self.beta)*self.input[i][1]
        delta_a=delta_a/len(self.input)
        delta_b=delta_b/len(self.input)
        self.alpha=self.alpha+learning_rate_alpha*delta_a
        self.beta=self.beta+learning_rate_beta*delta_b
    
    def getting_cost(self):
        cost=0
        for i in range(len(self.input)):
            cost=cost+(self.obtain[i]-self.alpha*np.power(self.input[i][0],self.beta)*
                       self.input[i][1])**2
        cost=cost/len(self.input)
        return cost 
    

def value(alpha,beta,X):
    return alpha*np.power(X[0],beta)*X[1]

def readCAR(file):
    ap=[[0.,0.,0.] for i in range(3)]
    atom_num=[]
    num=0
    fft_num=[]
    fft=[]
    
    fid=open(file,"rt")
    fid.readline()
    fid.readline()
    
    for i in range(3):
        line = fid.readline()
        for j in range(3):
            ap[i][j]=float(line.split()[j])
    
    fid.readline()
    line=fid.readline()
    str_num=line.split()
    
    for i in range(len(str_num)):
        atom_num.append(int(str_num[i]))
        num=num+atom_num[i]
        
    while not 'Direct' in line:
        line = str(fid.readline())
       
    for i in range(num):
        fid.readline()
    
    fid.readline()
    
    line=fid.readline()

    for i in range(len(line.split())):
        fft_num.append(int(line.split()[i]))
        
    line=fid.read()
    fft_str=line.split()
    for k in range(fft_num[2]):
        fft.append([])
        for j in range(fft_num[1]):
            fft[k].append([])
            for i in range(fft_num[0]):
                fft[k][j].append(float(fft_str[k*fft_num[1]*fft_num[0]+j*fft_num[0]+i]))
    fid.close()
    return ap,fft

def mean_pool(center,pool_num,X):
    pool_num=int(pool_num)
    pool_results=0
    for i in range(int(center[0]-pool_num/2),int(center[0]+pool_num/2+1)):
        for j in range(int(center[1]-pool_num/2),int(center[1]+pool_num/2+1)):
            pool_results=pool_results+X[i][j]            
    return pool_results/(pool_num*pool_num)
                
def mean_z(Rou,Elf,volume,pool_z):
    #原数据为[z][y][x]，返回的为[x][y]
    Rou_sz=[]
    Elf_sz=[]
    for i in range(len(Rou[0][0])):
        Rou_sz.append([])
        Elf_sz.append([])
        for j in range(len(Rou[0])):
            rou_sum=0
            elf_sum=0
            for k in range(pool_z[0],pool_z[1]):
                rou_sum=rou_sum+Rou[k][j][i]
                elf_sum=elf_sum+Elf[k][j][i]
            Rou_sz[i].append(rou_sum/((pool_z[1]-pool_z[0]-1)*volume))
            Elf_sz[i].append(elf_sum/(pool_z[1]-pool_z[0]-1))
    return Rou_sz,Elf_sz

def get_volume(ap):
    return np.inner(ap[2],np.cross(ap[0],ap[1]))

import numpy as np
import matplotlib.pyplot as plt
ap,Rou=readCAR('CHGCAR')
_,Elf=readCAR('ELFCAR')
print("Reading Finished")
pool_xyn=5
pool_z=[159,209]

volume=get_volume(ap)
#Center=[[81,107],[98,117],[113,106]]
Center=[[98,117],[113,106]]
X=[]
#Y=[2.6414,1.7232,1.1946]
Y=[1.7232,1.1946]

Rou_sz,Elf_sz=mean_z(Rou,Elf,volume,pool_z)
            

for i in range(len(Center)):
    X.append([])
    X[i].append(mean_pool(Center[i],pool_xyn,Elf_sz))
    X[i].append(mean_pool(Center[i],pool_xyn,Rou_sz)/10)
    Y[i]=Y[i]
    
print("Pooling Finished")
print("Begin to Fit")
AFMfit=Gra_fit(X,Y,a=0.01,b=0.01)
for i in range(80000):
    if i <= 20000:
        AFMfit.optimizer(learning_rate_alpha=4e-1,learning_rate_beta=4e-1)
    elif i<=50000:
        AFMfit.optimizer(learning_rate_alpha=1e-1,learning_rate_beta=1e-1)
    else:
        
        AFMfit.optimizer(learning_rate_alpha=5e-3,learning_rate_beta=5e-3)
#    print(AFMfit.getting_cost(),value(AFMfit.alpha,AFMfit.beta,X[0]),Y[0],
#          value(AFMfit.alpha,AFMfit.beta,X[1]),Y[1],
#          value(AFMfit.alpha,AFMfit.beta,X[2]),Y[2])
    print(AFMfit.getting_cost(),value(AFMfit.alpha,AFMfit.beta,X[0]),Y[0],
          value(AFMfit.alpha,AFMfit.beta,X[1]),Y[1])

print("Begin to print graph")
fit_res=[]
for i in range(len(Rou_sz)):
    fit_res.append([])
    for j in range(len(Rou_sz[i])):
        fit_res[i].append(value(AFMfit.alpha,AFMfit.beta,[Elf_sz[i][j],Rou_sz[i][j]/1e2]))
        
x=np.linspace(0,13.235700,len(fit_res))
y=np.linspace(0,12.736000,len(fit_res[i]))
x_mesh,y_mesh=np.meshgrid(x,y) 
c=plt.contourf(x_mesh,y_mesh,fit_res, 8, alpha = 1,cmap=plt.cm.afmhot)
