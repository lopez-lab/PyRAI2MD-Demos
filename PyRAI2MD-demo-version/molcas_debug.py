import numpy as np
import os,subprocess,shutil,h5py

def Printcoord(xyz):
    ## This function convert a numpy array of coordinates to a formatted string

    coord=''
    for line in xyz:
        e,x,y,z=line
        coord+='%-5s%16.8f%16.8f%16.8f\n' % (e,float(x),float(y),float(z))

    return coord

def S2F(M):
    ## This function convert 1D string (e,x,y,z) list to 2D float array

    M=[[float(x) for x in row.split()[1:4]] for row in M]
    return M

def whatistime():
    ## This function return current time

    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

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

def read_molcas(title):
    with open('%s.log' % (title),'r') as out:
        log  = out.read().splitlines()
        h5data   = h5py.File('%s.rasscf.h5' % (title),'r')
        natom    = 12
        casscf   = []
        gradient = []
        nac      = []

        civec    = np.array(h5data['CI_VECTORS'][()]) 
       	movec    = np.array(h5data['MO_VECTORS'][()])
        inactive = 0
        active   = 0
        for i,line in enumerate(log):
            if   """::    RASSCF root number""" in line:
                e=float(line.split()[-1])
                casscf.append(e)
            elif """Molecular gradients """ in line:
                g=log[i+8:i+8+natom]
                g=S2F(g)
                gradient.append(g)
            elif """CI derivative coupling""" in line:
                n=log[i+8:i+8+natom]
                n=S2F(n)
                nac.append(n)
            elif """Inactive orbitals""" in line and inactive == 0:
                inactive=int(line.split()[-1])
            elif """Active orbitals""" in line and active == 0:
                active=int(line.split()[-1])

        energy   = np.array(casscf)
        gradient = np.array(gradient)
        nac      = np.array(nac)
        norb     = int(len(movec)**0.5)
        movec    = movec[inactive*norb:(inactive+active)*norb]
        movec    = np.array(movec).reshape([active,norb])

    print(energy)
    print(len(gradient))
    print(len(nac))


read_molcas('2bt')
