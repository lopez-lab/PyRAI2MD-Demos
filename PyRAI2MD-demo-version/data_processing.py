### Data processing for ML models
### Jun 4 2020 Jingbai Li adapt this to PyRAIMD

import numpy as np
import json

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

def Prepdata(data,ml_seed,ratio,increment):
    data_info=''
    natom,nstate,xyz,invr,energy,gradient,nac,ci,mo=data
    xyz=np.array(xyz)
    invr=np.array(invr)
    energy=np.array(energy)
    gradient=np.array(gradient)
    nac=np.array(nac)

    nmol=len(invr)                    # number of molecule
    ninvr=len(invr[0])                # number of distance per molecule, which is the input size
    nenergy=len(energy[0])            # number of energy per molecule, which is the output size
    ngrad=len(gradient[0])/(natom*3)  # number of gradient matrix per molecule
    nnac=len(nac[0])/(natom*3)        # number of non-adiabatic matrix per molecule

    max_invr=np.amax(invr)
    min_invr=np.amin(invr)
    mid_invr=(max_invr+min_invr)/2
    dev_invr=(max_invr-min_invr)/2
    avg_invr=np.mean(invr)
    std_invr=np.std(invr)

    max_energy=np.amax(energy)
    min_energy=np.amin(energy)
    mid_energy=(max_energy+min_energy)/2
    dev_energy=(max_energy-min_energy)/2
    avg_energy=np.mean(energy)
    std_energy=np.std(energy)

    max_gradient=np.amax(gradient)
    min_gradient=np.amin(gradient)
    mid_gradient=(max_gradient+min_gradient)/2
    dev_gradient=(max_gradient-min_gradient)/2
    avg_gradient=np.mean(gradient)
    std_gradient=np.std(gradient)

    max_nac=np.amax(nac)
    min_nac=np.amin(nac)
    mid_nac=(max_nac+min_nac)/2
    dev_nac=(max_nac-min_nac)/2
    avg_nac=np.mean(nac)
    std_nac=np.std(nac)

    data_info+="""
  &training data
-------------------------------------------------------
  1/dist     max/min: %16.8f %16.8f
             avg/std: %16.8f %16.8f
             mid/dev: %16.8f %16.8f

  energy     max/min: %16.8f %16.8f
             avg/std: %16.8f %16.8f
       	     mid/dev: %16.8f %16.8f

  gradient   max/min: %16.8f %16.8f
             avg/std: %16.8f %16.8f
       	     mid/dev: %16.8f %16.8f

  nac        max/min: %16.8f %16.8f
             avg/std: %16.8f %16.8f
       	     mid/dev: %16.8f %16.8f

""" % (np.amax(invr),     np.amin(invr),     avg_invr,     std_invr,     mid_invr,     dev_invr,    \
       np.amax(energy),   np.amin(energy),   avg_energy,   std_energy,   mid_energy,   dev_energy,  \
       np.amax(gradient), np.amin(gradient), avg_gradient, std_gradient, mid_gradient, dev_gradient,\
       np.amax(nac),      np.amin(nac),      avg_nac,      std_nac,      mid_nac,      dev_nac)

#    This will be done by ML models 
#    # shift input to the averaged value and scaled by standard deviation
#    invr=(invr-miu_invr)/sgm_invr
#    energy=(energy-miu_energy)/sgm_energy
#    gradient=(gradient-miu_gradient)/sgm_gradient
#    nac=(nac-miu_nac)/sgm_nac
#    data_info+="""
#  &preprocessing data
#-------------------------------------------------------
#  1/dist     max/min: %16.8f %16.8f
#  energy     max/min: %16.8f %16.8f
#  gradient   max/min: %16.8f %16.8f
#  nac        max/min: %16.8f %16.8f
#""" % (np.amax(invr),     np.amin(invr),     np.amax(energy), np.amin(energy),\
#       np.amax(gradient), np.amin(gradient), np.amax(nac),    np.amin(nac))

    invr_train,invr_val,invr_test=partition(ml_seed,invr,ratio,increment)
    energy_train,energy_val,energy_test=partition(ml_seed,energy,ratio,increment)
    gradient_train,gradient_val,gradient_test=partition(ml_seed,gradient,ratio,increment)
    nac_train,nac_val,nac_test=partition(ml_seed,nac,ratio,increment)

    data_info+="""
  &post data
-------------------------------------------------------
  ml_seed: %8d
  ratio      train/validation/test: %5.2f %5.2f %5.2f
  dist       train/validation/test: %5d %5d %5d
  energy     train/validation/test: %5d %5d %5d
  gradient   train/validation/test: %5d %5d %5d
  nac        train/validation/test: %5d %5d %5d
""" % (ml_seed, ratio[0],            ratio[1],          1-ratio[0]-ratio[1],\
                len(invr_train),     len(invr_val),     len(invr_test),     \
                len(energy_train),   len(energy_val),   len(energy_test),   \
                len(gradient_train), len(gradient_val), len(gradient_test), \
                len(nac_train),len(nac_val),len(nac_test))

    postdata={
    'natom'          : natom,
    'nstate'         : nstate,
    'npair'          : int(nstate*(nstate-1)/2),
    'xyz'            : xyz,
    'invr'           : invr,
    'mean_invr'      : avg_invr,
    'std_invr'       : std_invr,
    'mid_invr'       : mid_invr,
    'dev_invr'       : dev_invr,
    'invr_train'     : invr_train,
    'invr_val'       : invr_val,
    'invr_test'      : invr_test,
    'energy'         : energy,
    'mean_energy'    : avg_energy,
    'std_energy'     : std_energy,
    'mid_energy'     : mid_energy,
    'dev_energy'     : dev_energy,
    'energy_train'   : energy_train,
    'energy_val'     : energy_val,
    'energy_test'    : energy_test,
    'gradient'       : gradient,
    'mean_gradient'  : avg_gradient,
    'std_gradient'   : std_gradient,
    'mid_gradient'   : mid_gradient,
    'dev_gradient'   : dev_gradient,
    'gradient_train' : gradient_train,
    'gradient_val'   : gradient_val,
    'gradient_test'  : gradient_test,
    'nac'            : nac,
    'mean_nac'       : avg_nac,
    'std_nac'        : std_nac,
    'mid_nac'        : mid_nac,
    'dev_nac'        : dev_nac,
    'nac_train'      : nac_train,
    'nac_val'        : nac_val,
    'nac_test'       : nac_test,
    }

    return postdata,data_info

def partition(sd,data,ratio,increment):
    np.random.seed(sd)
    size=len(data)
    pick_train=[]
    pick_validation=[]
    pick_test=[]
    weight_train=ratio[0]
    weight_validation=ratio[1]

    if increment == 0:
        block=1
        increment=size
    else:
        block=int(size/increment)

    for i in range(block):
        full=np.arange(size)[i*increment:(i+1)*increment]
        pick=np.random.choice(full,int(increment*weight_train),replace=False)
        pick_train=np.append(pick_train,pick)
        remain=[i for i in full if i not in pick_train]
        pick=np.random.choice(remain,int(increment*weight_validation),replace=False)
        pick_validation=np.append(pick_validation,pick)
        pick=[i for i in remain if i not in pick_validation]
        pick_test=np.append(pick_test,pick)

    train=data[pick_train.astype(int),:]
    validation=data[pick_validation.astype(int),:]
    test=data[pick_test.astype(int),:]

    return train,validation,test

def TrainDataInfo(variables):
    ## This function read train data and compute sgm (std or deviation) and miu (average or middle) and more
    ## This function is only used for interfacing with PyRAIMD
    
    in_data=variables['train_data']
    ml_seed=variables['ml_seed']
    ratio=variables['ratio']
    increment=variables['increment']
    with open('%s' % in_data,'r') as indata:
        data=json.load(indata)
    postdata,data_info=Prepdata(data,ml_seed,ratio,increment)

    return data,postdata,data_info

def AddTrainData(variables,newdata,iter):
    ## This function merge train data and new data then compute sgm (std or deviation) and miu (average or middle) and more
    ## This function is only used for interfacing with PyRAIMD

    data=variables['data']
    ml_seed=variables['ml_seed']
    ratio=variables['ratio']
    natom,nstate,xyz,invr,energy,gradient,nac,ci,mo=data
    for new in newdata:
        xyz     += [new[0]]
        invr    += [GetInvR(np.array(new[0])[:,1:4].astype(float)).tolist()]
        energy  += [new[1]]
        gradient+= [new[2]]
        nac     += [new[3]]
        ci      += [new[4]]
        mo      += [new[5]]
    data=[natom,nstate,xyz,invr,energy,gradient,nac,ci,mo]
    postdata,data_info=Prepdata(data,ml_seed,ratio,0)

    with open('New-data%s-%s.json' % (len(invr),iter),'w') as in_data:
        json.dump(data,in_data)

    return data,postdata,data_info

