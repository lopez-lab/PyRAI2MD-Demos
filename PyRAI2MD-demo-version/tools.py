## tools for PyRAIMD
## Jingbai Li Feb 13 2020
## Jingbai Li May 15 2020 add readinitcond

import os
import time,datetime,json
from periodic_table import Element
import numpy as np

def whatistime():
    ## This function return current time

    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

def howlong(start,end):
    ## This function calculate time between start and end

    walltime=end-start
    walltime='%5d days %5d hours %5d minutes %5d seconds' % (int(walltime/86400),int((walltime%86400)/3600),int(((walltime%86400)%3600)/60),int(((walltime%86400)%3600)%60))
    return walltime

def NACpairs(ci):
    ## This function generate a dictionary for non-adiabatic coupling pairs

    pairs={}
    n=0
    for i in range(ci):
        for j in range(i+1,ci):
            n+=1
            pairs[n]=[i+1,j+1]
            pairs[str([i+1,j+1])]=n
       	    pairs[str([j+1,i+1])]=n
    return pairs

def S2F(M):
    ## This function convert 1D string (e,x,y,z) list to 2D float array

    M=[[float(x) for x in row.split()[1:4]] for row in M]
    return M

def C2S(M):
    ## This function convert 2D complex array to 2D string array

    M=[[str(x) for x in row] for row in M]
    return M

def S2C(M):
    ## This function convert 2D string array back to 2D complex array

    M=[[complex(x) for x in row] for row in M]
    return M

def Readcoord(title):
    ## This function read coordinates from a xyz file
    ## This function return coordinates in a numpy array
    ## The elements are presented by the nuclear number
    ## This function also return a list of atomic mass in amu
    ## 1 g/mol = 1822.8852 amu

    file=open('%s.xyz'% (title)).read().splitlines()
    natom=int(file[0])
    xyz=[]
    mass=np.zeros((natom,1))
    for i,line in enumerate(file[2:2+natom]):
        e,x,y,z=line.split()
        xyz.append([e,x,y,z])
        m=Element(e).getMass()
        mass[i,0:1]=m*1822.8852

    return xyz,mass

def Readinitcond(trvm):
    ## This function read coordinates from sampled initial condition
    ## This function return coordinates in a numpy array
    ## The elements are presented by the nuclear number
    ## This function also return a list of atomic mass in amu
    ## 1 g/mol = 1822.8852 amu
    natom=len(trvm)
    xyz=[]
    velo=np.zeros((natom,3))
    mass=np.zeros((natom,1))
    for i,line in enumerate(trvm):
        e,x,y,z,vx,vy,vz,m,chrg=line
        xyz.append([e,x,y,z])
        m=Element(e).getMass()
        velo[i,0:3]=float(vx),float(vy),float(vz)
        mass[i,0:1]=float(m)*1822.8852
    
    return xyz,mass,velo    

def Printcoord(xyz):
    ## This function convert a numpy array of coordinates to a formatted string

    coord=''
    for line in xyz:
        e,x,y,z=line
        coord+='%-5s%24.16f%24.16f%24.16f\n' % (e,float(x),float(y),float(z))

    return coord

def Markatom(xyz,marks,prog):
    ## This function marks atoms for different basis set specification of Molcas

    new_xyz=[]

    for n,line in enumerate(xyz):
        e,x,y,z=line
        e = marks[n].split()[0]
        new_xyz.append([e,x,y,z])

    return new_xyz

def GetInvR(R):
    ## This functoin convert coordinates to inverse R matrix
    ## This function is only used for interfacing with PyRAIMD

    invr=[]
    q=R[1:]
    for atom1 in R:
        for atom2 in q:
            d=np.sum((atom1-atom2)**2)**0.5
            invr.append(1/d)
        q=q[1:]

    invr=np.array(invr)
    return invr

def Read_angle_index(var,n):
    ## This function read angle index from input or a file
    ## n = 2, it is equivalent to bond index

    file=var[0]
    if os.path.exists(file) == True:
        angle_list=np.loadtxt(file)
    else:
        num_angle=int(len(var)/n)
        tot_index=int(num_angle*n)
        angle_list=np.array(var[0:tot_index]).reshape([num_angle,n]).astype(int)

    if angle_list.shape[1] != n:
        print('The angle index file does not match the target angle')
        exit()

    angle_list-=1 # shift index starting from 0
    angle_list=angle_list.astype(int)
    return angle_list.tolist()

def PermuteMap(x,y_dict,permute_map,val_split):
    ## This function permute data following the map P.
    ## x is M x N x 3, M entries, N atoms, x,y,z
    ## y_dict has possible two keys 'energy_gradient' and 'nac'
    ## energy is M x n, M entries, n states
    ## gradient is M x n x N x 3, M entries, n states, N atoms, x,y,z
    ## nac is M x m x N x 3, M entries, m state pairs, N atoms, x,y,z
    ## permute_map is a file including all permutation

    # early stop the function
    if permute_map == 'No':
        return x, y_dict
    if permute_map != 'No' and os.path.exists(permute_map) == False:
       	return x, y_dict

    # load permutation map
    P = np.loadtxt(permute_map)-1
    P = P.astype(int)
    if len(P.shape) == 1:
        P = P.reshape([1,-1])

    x_new=np.zeros([0,x.shape[1],x.shape[2]]) # initialize coordinates list
    y_dict_new={}

    per_eg = 0
    per_nac = 0

    if 'energy_gradient' in y_dict.keys(): # check energy gradient
        energy = y_dict['energy_gradient'][0]  # pick energy, note permutation does not change energy
        grad = y_dict['energy_gradient'][1]    # pick gradient
        y_dict_new['energy_gradient'] = [np.zeros([0,energy.shape[1]]),np.zeros([0,grad.shape[1],grad.shape[2],grad.shape[3]])] # initialize energy and gradient list
        per_eg = 1

    if 'nac' in y_dict.keys():                 # check nac
       	nac = y_dict['nac']    	       	       # pick nac
        y_dict_new['nac'] = np.zeros([0,nac.shape[1],nac.shape[2],nac.shape[3]])                 # initialize nac list
        per_nac = 1

    kfold=np.ceil(1/val_split).astype(int)
    portion=int(len(x)*val_split)

    kfoldrange=[]
    for k in range(kfold):
        if k < kfold-1:
            kfoldrange.append([k*portion,(k+1)*portion])
        else:
            kfoldrange.append([k*portion,len(x)])

    for k in kfoldrange:
        # separate data in kfold
        a,b=k
        kx=x[a:b]
        new_x=kx
        if per_eg == 1:
            kenergy=energy[a:b]
            kgrad=grad[a:b]
            new_e=kenergy
            new_g=kgrad
        if per_nac == 1:
            knac = nac[a:b]
            new_n=knac

        for index in P:
            # permute coord along N atoms
            per_x = kx[:,index,:]
            new_x = np.concatenate([new_x,per_x],axis=0)
            if per_eg == 1:
                # permute grad along N atoms
                per_e = kenergy
                per_g = kgrad[:,:,index,:]              
                new_e = np.concatenate([new_e,per_e],axis=0)
                new_g = np.concatenate([new_g,per_g],axis=0)
            if per_nac == 1:
                # permute nac along N atoms
                per_n = knac[:,:,index,:]
                new_n = np.concatenate([new_n,per_n],axis=0)

        # merge the new data
        x_new=np.concatenate([x_new,new_x],axis=0)
        if per_eg == 1:   
            y_dict_new['energy_gradient'][0]=np.concatenate([y_dict_new['energy_gradient'][0],new_e],axis=0)
            y_dict_new['energy_gradient'][1]=np.concatenate([y_dict_new['energy_gradient'][1],new_g],axis=0)
        if per_nac == 1:
            y_dict_new['nac']=np.concatenate([y_dict_new['nac'],new_n],axis=0)

    return x_new,y_dict_new
  
