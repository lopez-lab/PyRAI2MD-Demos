### Neural Network for Photochemical Reaction Prediction
### Version 0.0 Jingbai Li Nov 13 2019
### Version 0.1 Jingbai Li Nov 19 2019 add hybrid training module NNEG
### Version 0.2 Jingbai Li Nov 23 2019 reconstruct the code structure
### Version 0.3 Jingbai Li Dec  2 2019 minor fix, imporve random search protocal
### Version 0.3 Jingbai Li Dec  4 2019 add fully random search
### Version 0.3 Jingbai Li Apr  4 2020 add more output info (mean and deviation) in data process
### Version 0.3 Jingbai Li Apr  6 2020 fix bugs in random search for NN
### Version 0.3 Jingbai Li Apr 11 2020 fix bugs in random search space generation for float inputs
### May 15 2020 Jingbai Li adapt this to PyRAIMD

from optparse import OptionParser
import time,datetime,os,sys,json
import numpy as np
from data_processing import Prepdata
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation,Input,BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler

import talos as ta
from talos import Scan

def J2LA(L):
    ## This function load json file and convert to a list of numpy array
    ## This function is used to import weights for neural network
    ## If a json file is None, it returns L as None

    if L != None:
        with open('%s' % L,'r') as list_nparray:
            L=json.load(list_nparray)
            L=[np.array(i) for i in L]

    return L

def L2A(M):
    ## this function convert list of E, G, and N to numpy array as required by dyanmics
    E,G,N=M
    E=np.array(E).reshape([-1])
    states=len(E)
    pairs=int(states*(states-1)/2)
    G=np.array(G).reshape([states,-1,3])
    N=np.array(N).reshape([pairs,-1,3])

    return E,G,N

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

def whatistime():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

def howlong(start,end):
    walltime=end-start
    walltime='%5d days %5d hours %5d minutes %5d seconds' % (int(walltime/86400),int((walltime%86400)/3600),int(((walltime%86400)%3600)/60),int(((walltime%86400)%3600)%60))
    return walltime

def params_space(space,var,par):
    #span hyperparameter space

    start=np.amin(space[0:2])
    end=np.amax(space[0:2])
    step=space[2]
    params=[]

    if   step == 0:
        params=[var]
    elif step > 0:
        for i in range(step+1):
            if   par == 'epoch' or par == 'layer' or par == 'nodes' or par == 'flrstep':	   # (last-initial)/step
                p=start+(end-start)*(i/step)
                p=int(p)
            elif par == 'batch' or par == 'wl2' or par == 'lr' or par == 'flr':     # (last/initial)**(1/step)
                p=start*(end/start)**(i/step)
                if par == 'batch':
                    p=int(p)
            if p not in params:
                params.append(p)
    elif step < 0:
        step=1+step*-1
        np.random.seed(int(time.time()%1/0.001))  # set a random seed for random search
        if   par == 'epoch' or par == 'layer' or par == 'nodes' or par == 'flrstep' or par == 'batch':           # random integer (last - initial)
            params=np.random.choice(end-start+1,np.amin([end-start+1,step]),replace=False)+start
        elif par == 'wl2' or par == 'lr' or par == 'flr':     # random float (0 - last/initial)
            if end/start > 100:
                base=1000
            else:
                base=100
            params=np.random.choice(base+1,int(step),replace=False)/base
            params=start*(end/start)**params
        params=sorted(params)
    return params

def update_params(candidates,space,par):
    start,end,step=space
    if step !=0:
        index={'batch':-11,'epoch':-10,'flr':-9,'flrstep':-8,'lr':-4,'layer':-5,'nodes':-3,'wl2':-1}
        candidates=candidates[:,index[par]]
        max=np.amax(candidates)
        min=np.amin(candidates)
        if par == 'batch' or par == 'epoch' or par == 'layer' or par == 'nodes' or par == 'flrstep':
            max=int(max)
            min=int(min)
        if max == min:
            space=[max,max,1]
        else:
            space=[min,max,step]

    return space

def lr_scheduler(epoch,lr):
    #flrstep and flr are global varable
    if (epoch+1) % flrstep == 0:
        lr=lr*flr
    return lr

def shifted_softplus(x):
    return K.log(0.5*K.exp(x)+0.5)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true), axis=-1))

def dmax(y_true,y_pred):
    return K.max(K.abs(y_pred -y_true))

def Record(chkname,model_name,arch,hist):

    name_list={
    'e'  : 'energy',
    'g'  : 'gradient',
    'nac': 'non-adiabatic coupling',
    }

    if 'eg' in model_name:
        output='%s\n' % (arch)
        output+='\n  &training history %s\n' % (name_list['e'])
        output+='-------------------------------------------------------\n'
        output+='Para%6s%16s%16s%16s%16s%16s%16s%16s%16s%16s\n' % ('epoch','lr','loss','val_loss','mae','val_mae','rmse','val_rmse','dmax','val_dmax')
        n=len(hist['lr'])
        for i in range(n):
            output+='Ep: %6d%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % ((i+1),hist['lr'][i],hist['e_loss'][i],hist['val_e_loss'][i],hist['e_mae'][i],hist['val_e_mae'][i],hist['e_rmse'][i],hist['val_e_rmse'][i],hist['e_dmax'][i],hist['val_e_dmax'][i])
        output+='\n  &training history %s\n' % (name_list['g'])
        output+='-------------------------------------------------------\n'
        output+='Para%6s%16s%16s%16s%16s%16s%16s%16s%16s%16s\n' % ('epoch','lr','loss','val_loss','mae','val_mae','rmse','val_rmse','dmax','val_dmax')
        n=len(hist['lr'])
        for i in range(n):
            output+='Ep: %6d%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % ((i+1),hist['lr'][i],hist['g_loss'][i],hist['val_g_loss'][i],hist['g_mae'][i],hist['val_g_mae'][i],hist['g_rmse'][i],hist['val_g_rmse'][i],hist['g_dmax'][i],hist['val_g_dmax'][i])
        output+='\n'
        log=open('%s.log' % (chkname),'a')
        log.write(output)
        log.close()
        result='%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % (hist['lr'][-1],hist['e_loss'][-1],hist['val_e_loss'][-1],hist['e_mae'][-1],hist['val_e_mae'][-1],hist['e_rmse'][-1],hist['val_e_rmse'][-1],hist['e_dmax'][-1],hist['val_e_dmax'][-1])
        result+='%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % (hist['lr'][-1],hist['g_loss'][-1],hist['val_g_loss'][-1],hist['g_mae'][-1],hist['val_g_mae'][-1],hist['g_rmse'][-1],hist['val_g_rmse'][-1],hist['g_dmax'][-1],hist['val_g_dmax'][-1])
        fin=open('%s.sum' % (chkname),'a')
        fin.write(result)
        fin.close()
    else:
        output='%s\n' % (arch)
        output+='  &training history %s\n' % (name_list[model_name])
        output+='-------------------------------------------------------\n'
        output+='Para%6s%16s%16s%16s%16s%16s%16s%16s%16s%16s\n' % ('epoch','lr','loss','val_loss','mae','val_mae','rmse','val_rmse','dmax','val_dmax')
        n=len(hist['lr'])
        for i in range(n):
            output+='Ep: %6d%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % ((i+1),hist['lr'][i],hist['loss'][i],hist['val_loss'][i],hist['mae'][i],hist['val_mae'][i],hist['rmse'][i],hist['val_rmse'][i],hist['dmax'][i],hist['val_dmax'][i],)
        output+='\n'
        log=open('%s.log' % (chkname),'a')
        log.write(output)
        log.close()
        result='%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n' % (hist['lr'][-1],hist['loss'][-1],hist['val_loss'][-1],hist['mae'][-1],hist['val_mae'][-1],hist['rmse'][-1],hist['val_rmse'][-1],hist['dmax'][-1],hist['val_dmax'][-1])
        fin=open('%s.sum' % (chkname),'a')
        fin.write(result)
        fin.close()

def NNEG(feat_train,target_train,feat_val,target_val,params):

    epoch=params['epoch']
    batch=params['batch']
    layer=params['layer']
    nodes=params['nodes']
    wl2=params['wl2']
    lr=params['lr']
    flr=params['flr']
    flrstep=params['flrstep']
    in_weight=params['in_weight']
    import_weights=params['import_weights']
    silent=params['silent']

    e_train,g_train=target_train
    e_val,g_val=target_val

    dim_in=len(feat_train[0])
    dim_out_e=len(e_train[0])
    dim_out_g=len(g_train[0])

    ## input layer
    input=Input(shape=(dim_in,))
    dense_e=Dense(nodes,kernel_regularizer=regularizers.l2(wl2),activation='tanh')(input)
    dense_e=BatchNormalization()(dense_e)
    dense_g=Dense(nodes,kernel_regularizer=regularizers.l2(wl2),activation='tanh')(input)
    dense_g=BatchNormalization()(dense_g)

    ## hidden layers
    for hd in range(layer):
        dense_e=Dense(nodes,kernel_regularizer=regularizers.l2(wl2),activation='tanh')(dense_e)
        dense_e=BatchNormalization()(dense_e)
        dense_g=Dense(nodes,kernel_regularizer=regularizers.l2(wl2),activation='tanh')(dense_g)
        dense_g=BatchNormalization()(dense_g)

    ## output layer
    dense_e=Dense(dim_out_e,kernel_regularizer=regularizers.l2(wl2),activation='linear',name='e')(dense_e)
    dense_g=Dense(dim_out_g,kernel_regularizer=regularizers.l2(wl2),activation='linear',name='g')(dense_g)

    model=Model(inputs=input,outputs=[dense_e,dense_g])
    model._name="double"
    target_train_dict={'e':e_train,'g':g_train}
    target_val_dict={'e':e_val,'g':g_val}
    adam = ks.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
    optimizer=adam,
    loss={'e':'mean_squared_error','g':'mean_squared_error'},
    loss_weights={'e':0.5,'g':0.5},
    metrics={'e':['mae',rmse,dmax],'g':['mae',rmse,dmax]}
    )

    if silent == 0:
        print(model.summary())

    if in_weight == -1:
        model.set_weights(import_weights)
        print('Successfully load weights')
#        model.load_weights('%s-%s' % (model_name,weights_h5))
        history = model.predict(
        feat_train
        )
    else:
        if in_weight >0:
            model.set_weights(import_weights)
            print('Successfully load weights')
#            model.load_weights('%s-%s' % (model_name,weights_h5))
        history = model.fit(
        feat_train,
        target_train,
        epochs=epoch,
        batch_size=batch,
        callbacks=[ks.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)],
        validation_data=[feat_val,target_val],
        shuffle=True
        )

    return history, model

def NN(feat_train,target_train,feat_val,target_val,params):
    epoch=params['epoch']
    batch=params['batch']
    layer=params['layer']
    nodes=params['nodes']
    wl2=params['wl2']
    lr=params['lr']
    flr=params['flr']
    flrstep=params['flrstep']
    in_weight=params['in_weight']
    import_weights=params['import_weights']
    silent=params['silent']

    dim_in=len(feat_train[0])
    dim_out=len(target_train[0])
    ## input layer
    model = Sequential([
      Dense(nodes, input_shape=(dim_in,),kernel_regularizer=regularizers.l2(wl2),activation='tanh'),
      BatchNormalization()
    ])
    model._name="single"
    ## hidden layers
    for hd in range(layer):
        model.add(Dense(nodes,kernel_regularizer=regularizers.l2(wl2),activation='tanh'))
        model.add(BatchNormalization())
    ## output layer
    model.add(Dense(dim_out,kernel_regularizer=regularizers.l2(wl2),activation='linear'))

    adam = ks.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
    optimizer=adam,
    loss='mean_squared_error',
    metrics=['mae',rmse,dmax],
    )

    if silent == 0:
        print(model.summary())

    if in_weight == -1:
        model.set_weights(import_weights)
        print('Successfully load weights')
#        model.load_weights('%s-%s' % (model_name,weights_h5))
        history = model.predict(
        feat_train
        )
    else:
        if in_weight >0:
            model.set_weights(import_weights)
            print('Successfully load weights')
#            model.load_weights('%s-%s' % (model_name,weights_h5))
        history = model.fit(
        feat_train,
        target_train,
        epochs=epoch,
        batch_size=batch,
        callbacks=[ks.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)],
        validation_data=[feat_val,target_val],
        shuffle=True
        )

    return history, model


def RunNN(title,T,invR,variables,in_weight,group):
    ## This function run NNs
    return_data=''

    start=time.time()

    heading="""
 *--------------------------------------------------------------*
 |                                                              |
 |   Neural Network for Prediciton of Photochemical Reaction    |
 |                                                              |
 *--------------------------------------------------------------*

"""
    topline='Neural Network Start: %20s\n%s' % (whatistime(),heading)

    model_name_list={
    'eg1' :'eg',
    'eg2' :'eg',
    'e1'  :'e',
    'e2'  :'e',
    'g1'  :'g',
    'g2'  :'g',
    'nac1':'nac',
    'nac2':'nac',
    }

    ## unpack training data
    postdata    = variables['postdata']
    pred_data   = variables['pred_data']
    data_info   = variables['data_info']
    ml_seed     = variables['ml_seed']
    silent	= variables['silent']
    target      = variables['target']
    model_name  = model_name_list[target]

    ## upack model variables
    global flr,flrstep
    epoch   = variables[target]['epoch']
    batch   = variables[target]['batch']
    layer   = variables[target]['layer']
    nodes   = variables[target]['nodes']
    wl2     = variables[target]['wl2']
    lr      = variables[target]['lr']
    flr     = variables[target]['flr']
    flrstep = variables[target]['flrstep']
    nsample = variables[target]['nsample']
    space_epoch   = variables[target]['s_epoch']
    space_batch   = variables[target]['s_batch']
    space_layer   = variables[target]['s_layer']
    space_nodes   = variables[target]['s_nodes']
    space_wl2     = variables[target]['s_wl2']
    space_lr      = variables[target]['s_lr']
    space_flr     = variables[target]['s_flr']
    space_flrstep = variables[target]['s_flrstep']
    s_iter        = variables[target]['s_iter']
    s_win         = variables[target]['s_win']
    weights_list  = variables[target]['weights_list']

    ## translate to NN params variables
    params={
    'epoch'         : epoch,
    'batch'         : batch,
    'layer'         : layer,
    'nodes'         : nodes,
    'wl2'           : wl2,
    'lr'            : lr,
    'flr'           : flr,
    'flrstep'       : flrstep,
    'in_weight'     : in_weight,
    'silent'        : silent,
    'import_weights': weights_list,
    }

    ## initialize NN models
    model_list={
    'eg' : NNEG,
    'e'  : NN,
    'g'  : NN,
    'nac': NN,
    }

    invr_train=postdata['invr_train']
    invr_val=postdata['invr_val']

    y_list={
    'eg' :[postdata['energy_train'],postdata['gradient_train']],
    'e'  : postdata['energy_train'],
    'g'  : postdata['gradient_train'],
    'nac': postdata['nac_train']
    }

    y_val_list={
    'eg' :[postdata['energy_val'],postdata['gradient_val']],
    'e'  : postdata['energy_val'],
    'g'  : postdata['gradient_val'],
    'nac': postdata['nac_val'],
    }

    sgm_list={
    'invr': postdata['sgm_invr'],
    'e'   : postdata['sgm_energy'],
    'g'   : postdata['sgm_gradient'],
    'nac' : postdata['sgm_nac'],
    }

    miu_list={
    'invr': postdata['miu_invr'],
    'e'   : postdata['miu_energy'],
    'g'   : postdata['miu_gradient'],
    'nac' : postdata['miu_nac'],
    }

    ## print and save output before run NN
    run_info="""
  &nn run mode %d
-------------------------------------------------------
   -2 Hyperparameter search |   0 New train
   -1 Prediction            |  >0 Load weights
""" % (in_weight)

    if silent == 0:
        print(topline)
        print(data_info)
        print(run_info)

    if group == None:
        chkname='NN-%s-%s' % (title,model_name)
    else:
        chkname='NN-%s-%s-%s'   % (title,model_name,group)

    log=open('%s.log' % (chkname),'w')
    log.write(topline)
    log.write(data_info)
    log.write(run_info)
    log.close()
    fin=open('%s.sum' % (chkname),'w')
    fin.write('')
    fin.close()

    #run neural netwrok
    if   in_weight >= 0:  # New train or restart train

        history,model=model_list[model_name](invr_train,y_list[model_name],invr_val,y_val_list[model_name],params)

        train_info="""
  &start training nn
-------------------------------------------------------
    Model: %20s
    Epock: %20d
    Batch: %20d
    Layer: %20d
    Nodes: %20d
    Lrate: %20.16f
    Decay: %20.16f
    L2reg: %20.16f

  &model summary
-------------------------------------------------------
"""% (model_name,epoch,batch,layer,nodes,lr,flr,wl2)

        log=open('%s.log' % (chkname),'a')
        log.write(train_info)
        log.close()

        w=model.get_weights()
        wl=[i.tolist() for i in w]
        with open('%s.json' % (chkname),'w') as export_weights:
            json.dump(wl,export_weights)

#        model.save_weights('%s-%s' % (model_name,weights_h5))
        hist=history.history
        arch=[]
        model.summary(print_fn=lambda x: arch.append(x))
        arch= '\n'.join(arch)
        Record(chkname,model_name,arch,hist)
        return_data=w

    elif in_weight == -1: # Prediction
        if invR == None:
            with open('%s' % pred_data,'r') as preddata:
                pred=json.load(preddata)            
            pred_natom,pred_nstate,pred_xyz,pred_invr,pred_energy,pred_gradient,pred_nac,pred_ci=pred
        else:
            pred_invr=invR

        invr_train=(pred_invr-miu_list['invr'])/sgm_list['invr']
        history,model=model_list[model_name](invr_train,y_list[model_name],invr_val,y_val_list[model_name],params)

        if type(history) == list:  # when use eg model
            q=open('%s-e.pred.txt' % (chkname),'w')
       	    p=open('%s-g.pred.txt' % (chkname),'w')
            e=history[0]*sgm_list['e']+miu_list['e']
            g=history[1]*sgm_list['g']+miu_list['g']
            np.savetxt(q,e)
            np.savetxt(p,g)
            q.close()
            p.close()
            return_data=[e,g]
        
        elif type(history) == np.ndarray:  # when use e,g, or nac model
            q=open('%s.pred.txt' % (chkname),'w')
            e=history*sgm_list[model_name]+miu_list[model_name]
            np.savetxt(q,e)
            return_data=[e]

    elif in_weight == -2: # Hyperparameter search

        ## wrap hyperparameter space into a dictionory. p must have the same keys as params!
        p={
        'epoch'         : params_space(space_epoch,epoch,'epoch'),
        'batch'         : params_space(space_batch,batch,'batch'),
        'layer'         : params_space(space_layer,layer,'layer'),
        'nodes'         : params_space(space_nodes,nodes,'nodes'),
        'wl2'           : params_space(space_wl2,wl2,'wl2'),
        'lr'            : params_space(space_lr,lr,'lr'),
        'flr'           : params_space(space_flr,flr,'flr'),
        'flrstep'       : params_space(space_flrstep,flrstep,'flrstep'),
        'import_weights':[None],
        'in_weight'     :[in_weight],
        'silent'        :[silent],
        }

        for i in range(s_iter):
            Permut=len(p['epoch'])*len(p['batch'])*len(p['layer'])*len(p['nodes'])*len(p['lr'])*len(p['flr'])*len(p['flrstep'])*len(p['wl2'])

            if   int(Permut) >= s_win and int(nsample*Permut) <= s_win:
                nsample=float(s_win+0.0001)/Permut  # increase sample ratio, add 0.0001 to round ratio up
            elif int(Permut) <  s_win:
                nsample=1                    # search all space

            search_info="""
  &search space %d
-------------------------------------------------------
    Epock: %50s
    Batch: %50s
    Layer: %50s
    Nodes: %50s
    Lrate: %50s
    Decay: %50s
    Dwait: %50s
    L2reg: %50s
    Ratio: %50s
    Total: %50s
   Sample: %50s
   Window: %50s
""" % (i+1,p['epoch'],p['batch'],p['layer'],p['nodes'],p['lr'],p['flr'],p['flrstep'],p['wl2'],nsample,Permut,int(nsample*Permut),s_win)
            log=open('%s.log' % (chkname),'a')
            log.write(search_info)
            log.close()

            if silent == 0:
                print(search_info)

            if int(nsample*Permut) == 1:
                break                        # not enough space to search

            h = ta.Scan(x=invr_train,y=y_list[model_name],x_val=invr_val,y_val=y_val_list[model_name],
            model=model_list[model_name],params=p,experiment_name=model_name,
            #reduction_method='', reduction_metric='val_mae',
            random_method='quantum',seed=ml_seed,fraction_limit=nsample)

            candidates=np.array(h.data)
            candidates=candidates[np.argsort(candidates[:,1])]   # sort as val_loss
            candidates=candidates[0:s_win]                       # select candidates
            if i == 0:
                candidates_group=np.copy(candidates) #generate a group of candidates
            else:
                candidates_group=np.concatenate((candidates_group,candidates))  # add candidates to group
                candidates=candidates_group[np.argsort(candidates_group[:,1])]  # sort group as val_loss
                candidates=candidates[0:s_win]                                  # select candidates

            print(h.data)
            search_info="""
  &search results %d
-------------------------------------------------------
  Candidates: %6d Group: %6d
  %5s%6s%6s%6s%6s%20s%20s%6s%20s%16s
""" % (i+1,len(candidates),len(candidates_group),'No.','epoch','batch','layer','nodes','rate','decay','wait','L2reg','val_loss')
            for j in range(s_win):
                #index={'batch':-11,'epoch':-10,'flr':-9,'flrstep':-8,'lr':-4,'layer':-5,'nodes':-3,'wl2':-1}
                search_info+='  %5d%6d%6d%6d%6d%20.16f%20.16f%6d%20.16f%16.8f\n' % (j+1,candidates[j][-10],candidates[j][-11],candidates[j][-5],candidates[j][-3],candidates[j][-4],candidates[j][-9],candidates[j][-8],candidates[j][-1],candidates[j][1])

            search_info+='\n'
            log=open('%s.log' % (chkname),'a')
            log.write(search_info)
            log.close()

            if silent == 0:
                print(h.details)
                print(search_info)

            ## update hyperparameter space
            space_epoch=update_params(candidates,space_epoch,'epoch')
            space_batch=update_params(candidates,space_batch,'batch')
            space_layer=update_params(candidates,space_layer,'layer')
            space_nodes=update_params(candidates,space_nodes,'nodes')
            space_wl2=update_params(candidates,space_wl2,'wl2')
            space_lr=update_params(candidates,space_lr,'lr')
            space_flr=update_params(candidates,space_flr,'flr')
            space_flrstep=update_params(candidates,space_flrstep,'flrstep')

            ## update params dictionary
            p={
            'epoch'         : params_space(space_epoch,epoch,'epoch'),
            'batch'         : params_space(space_batch,batch,'batch'),
            'layer'         : params_space(space_layer,layer,'layer'),
            'nodes'         : params_space(space_nodes,nodes,'nodes'),
            'wl2'           : params_space(space_wl2,wl2,'wl2'),
            'lr'            : params_space(space_lr,lr,'lr'),
            'flr'           : params_space(space_flr,flr,'flr'),
            'flrstep'       : params_space(space_flrstep,flrstep,'flrstep'),
            'import_weights':[None],
            'in_weight'     :[in_weight],
            'silent'        :[silent]
            }


    end=time.time()
    walltime=howlong(start,end)
    endline='Neural Network End: %20s Total: %20s\n' % (whatistime(),walltime)

    log=open('%s.log' % (chkname),'a')
    log.write(endline)
    log.close()

    usage='%s\n' % (end-start)

    fin=open('%s.sum' % (chkname),'a')
    fin.write(usage)
    fin.close()

    if silent == 0:
        print('\n%s' % endline)

    return return_data

def NNPRED(title,T,R,method_variables):
    ## This function interface with PyRAIMD to predicts energy, gradient, and non-adiabtaic coupling 
    ## This function call RunNN
    C=np.array([])
    invR=[GetInvR(R)] # convert to inverse R

    ## unpack variables
    variables_eg1=method_variables['eg1']
    variables_eg2=method_variables['eg2']
    variables_e1=method_variables['e1']
    variables_e2=method_variables['e2']

    ## for general prediction
    if   variables_eg1['active'] == 1:
        method_variables['target']='eg1'
        [e1_pred,g1_pred]=RunNN(title,T,invR,method_variables,in_weight=-1,group=None)
    else:
        method_variables['target']='e1'
        e1_pred=RunNN(title,T,invR,method_variables, in_weight=-1,group=None)
        method_variables['target']='g1'
       	g1_pred=RunNN(title,T,invR,method_variables, in_weight=-1,group=None)
    method_variables['target']='nac1'
    nac1_pred=RunNN(title,T,invR,method_variables,   in_weight=-1,group=None)

    E,G,N=L2A([e1_pred,g1_pred,nac1_pred])
    RMSD=[None,None,None]

    ## for adaptive sampling
    if   variables_eg2['active'] == 1 or variables_e2['active'] == 1:
        if  variables_eg2['active'] == 1:
            method_variables['target']='eg2'
            [e2_pred,g2_pred]=RunNN(title,T,invR,method_variables,in_weight=-1,group=None)
        elif variables_e2['active'] == 1:
            method_variables['target']='e2'
            e2_pred=RunNN(title,T,invR,method_variables, in_weight=-1,group=None)
            method_variables['target']='g2'
            g2_pred=RunNN(title,T,invR,method_variables, in_weight=-1,group=None)
        method_variables['target']='nac2'
        nac2_pred=RunNN(title,T,invR,method_variables,   in_weight=-1,group=None)
        dE=np.array(e1_pred)-np.array(e2_pred)
        dG=np.array(g1_pred)-np.array(g2_pred)
        dN=np.array(nac1_pred)-np.array(nac2_pred)
        dE=np.mean(dE**2)**0.5
        dG=np.mean(dG**2)**0.5
        dN=np.mean(dN**2)**0.5
        RMSD=[dE,dG,dN]

    return E,G,N,C,RMSD

def NNTRAIN(title,method_variables,i):
    ## This function interface with PyRAIMD to train nn with new data
    ## This function call RunNN
    ## method_variables is expected to have 2 items in adpative sampling for nn

    T=[]
    invR=None

    ## unpack variables
    variables_eg1=method_variables['eg1']
    variables_eg2=method_variables['eg2']

    ## train the first group of NNs and update weights_list
    if variables_eg1['active'] == 1:
        method_variables['target']='eg1'
        method_variables['eg1']['weights_list']=RunNN(title,T,invR,method_variables,in_weight=0,group='a-%s' % (i))
    else:
       	method_variables['target']='e1'
        method_variables['e1']['weights_list']=RunNN(title,T,invR,method_variables, in_weight=0,group='a-%s' % (i))
       	method_variables['target']='g1'
        method_variables['g1']['weights_list']=RunNN(title,T,invR,method_variables, in_weight=0,group='a-%s' % (i))
    method_variables['target']='nac1'
    method_variables['nac1']['weights_list']=RunNN(title,T,invR,method_variables,   in_weight=0,group='a-%s' % (i))

    ## train the second group of NNs and update weights_list
    if variables_eg2['active'] == 1:
       	method_variables['target']='eg2'
        method_variables['eg2']['weights_list']=RunNN(title,T,invR,method_variables,in_weight=0,group='b-%s' % (i))
    else:
       	method_variables['target']='e2'
        method_variables['e2']['weights_list']=RunNN(title,T,invR,method_variables, in_weight=0,group='b-%s' % (i))
       	method_variables['target']='g2'
        method_variables['g2']['weights_list']=RunNN(title,T,invR,method_variables, in_weight=0,group='b-%s' % (i))
    method_variables['target']='nac2'
    method_variables['nac2']['weights_list']=RunNN(title,T,invR,method_variables,   in_weight=0,group='b-%s' % (i))

    return 0

def NNMODEL(variables_all,in_weight):
    ## This function interface with PyRAIMD to tune nn
    ## This function call RunNN

    T=[]
    invR=None
    title=variables_all['control']['title']
    prediction=RunNN(title,T,invR,variables_all['nn'],in_weight=in_weight,group=None)

def main():
    ## This is the main function for testing NN models
    ## This function call RunNN

    usage="""
 *--------------------------------------------------------------*
 |                                                              |
 |   Neural Network for Prediciton of Photochemical Reaction    |
 |                                                              |
 *--------------------------------------------------------------*

Usage:
  python3 NN-ChemI.py --td in_data [additional options]
  python3 NN-ChemI.py -h for more options
"""
    description=''
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('--tt', dest='title',     type=str,   nargs=1, help='Title of the model')
    parser.add_option('--td', dest='in_data',   type=str,   nargs=1, help='Training data in json format')
    parser.add_option('--pd', dest='pred_data', type=str,   nargs=1, help='Prediction data in json format')
    parser.add_option('--mr', dest='selc_energy', type=int,   nargs=1, help='Select energy. 0 all; 1 casscf; 2 caspt2. Default is 0.',default=0)
    parser.add_option('--sl', dest='silent',    type=int,   nargs=1, help='=0 silent mode; =1 print verbose information; Default=0',default=0)
    parser.add_option('--gs', dest='ml_seed',   type=int,   nargs=1, help='Global random seed; Defualt=0',default=0)
    parser.add_option('--iw', dest='in_weight', type=int,   nargs=1, help='Neural Network modes: -2 hyper parameter search, requres Talos; -1 predict properties; 0 new train; >0 load trained weights',default=0)
    parser.add_option('--ip', dest='import_weights', type=str,   nargs=1, help='Import weights',default=None) 
    parser.add_option('--st', dest='stat',      type=int,   nargs=1, help='Plot statistics of in_data, requres matplotlib; Defualt=0',default=0)
    parser.add_option('--nn', dest='model_name',type=str,   nargs=1, help='Type of neural network; eg - energy+gradient; nac - non-adiabatic coupling; e - energy; g - gradient', default='eg')
    parser.add_option('--ep', dest='epoch',     type=int,   nargs=1, help='Epoch; Default=1',default=1)
    parser.add_option('--bs', dest='batch',     type=int,   nargs=1, help='Batch size; Default=1',default=1)
    parser.add_option('--hl', dest='layer',     type=int,   nargs=1, help='Hidden layer; Default=1',default=1)
    parser.add_option('--nd', dest='nodes',     type=int,   nargs=1, help='Node per hidden layer; Default=1',default=1)
    parser.add_option('--l2', dest='wl2',       type=float, nargs=1, help='L2 regularization rate; Default=1e-9',default=1e-9)
    parser.add_option('--lr', dest='lr',        type=float, nargs=1, help='Learning rate; Default=3e-3',default=3e-3)
    parser.add_option('--dl', dest='flr',       type=float, nargs=1, help='Learning rate decay factor; Default=0.9',default=0.9)
    parser.add_option('--ds', dest='flrstep',   type=int,   nargs=1, help='Learning rate decay factor waiting step; Default=10',default=10)
    parser.add_option('--NS', dest='nsample',   type=float, nargs=1, help='Random search sample ratio, iteration=ratio*search_space_size; Default=0.1',default=0.1)
    parser.add_option('--EP', dest='s_epoch',   type=int,   nargs=3, help='Random search epoch, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
    parser.add_option('--BS', dest='s_batch',   type=int,   nargs=3, help='Random search batch size, requires initial, last and steps; Default=1 1 0',default=[1,1,0])
    parser.add_option('--HL', dest='s_layer',   type=int,   nargs=3, help='Random search hidden layer, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
    parser.add_option('--ND', dest='s_nodes',   type=int,   nargs=3, help='Random search node, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
    parser.add_option('--LL', dest='s_wl2',     type=float, nargs=3, help='Random search L2 regularization rate, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
    parser.add_option('--LR', dest='s_lr',      type=float, nargs=3, help='Random search learning rate, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
    parser.add_option('--DL', dest='s_flr',     type=float, nargs=3, help='Random search learning rate decay factor, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
    parser.add_option('--DS', dest='s_flrstep', type=int,   nargs=3, help='Random search learning rate decay waiting step, requires inital, last, and steps; Default=1 1 0',default=[1,1,0])
    parser.add_option('--NI', dest='s_iter',    type=int,   nargs=1, help='Random search iteractions; Default=1',default=1)
    parser.add_option('--WN', dest='s_win',     type=int,   nargs=1, help='Random search candidates; Defualt=4',default=4)

    (options, args) = parser.parse_args()
    if options.in_data == None:
        print (usage)
        exit()
    #np.random.seed(options.ml_seed)   # fix the random seed befor import keras

    title=options.title
    in_data=options.in_data
    pred_data=options.pred_data
    silent=options.silent
    ml_seed=options.ml_seed
    in_weight=options.in_weight
    import_weights=options.import_weights
    stat=options.stat
    model_name=options.model_name
    epoch=options.epoch
    batch=options.batch
    layer=options.layer
    nodes=options.nodes
    wl2=options.wl2
    lr=options.lr
    flr=options.flr
    flrstep=options.flrstep
    nsample=options.nsample
    space_epoch=options.s_epoch
    space_batch=options.s_batch
    space_layer=options.s_layer
    space_nodes=options.s_nodes
    space_wl2=options.s_wl2
    space_lr=options.s_lr
    space_flr=options.s_flr
    space_flrstep=options.s_flrstep
    s_iter=options.s_iter
    s_win=options.s_win


    with open('%s' % in_data,'r') as indata:
        data=json.load(indata)
    ratio=[0.9,0.1]
    postdata,data_info=Prepdata(data,ml_seed,ratio)

    
    variables_nn={
    'postdata'    :postdata,
    'pred_data'   :pred_data,
    'data_info'   :data_info,
    'silent'      :silent,
    'ml_seed'     :ml_seed,
    }

    variables={
    'epoch'	  :epoch,
    'batch'	  :batch,
    'layer'	  :layer,
    'nodes'	  :nodes,
    'wl2'         :wl2,
    'lr'          :lr,
    'flr'         :flr,
    'flrstep'     :flrstep,
    'nsample'     :nsample,
    's_epoch'     :space_epoch,
    's_batch'     :space_batch,
    's_layer'     :space_layer,
    's_nodes'     :space_nodes,
    's_wl2'	  :space_wl2,
    's_lr'        :space_lr,
    's_flr'	  :space_flr,
    's_flrstep'   :space_flrstep,
    's_iter'	  :s_iter,
    's_win'	  :s_win,
    'weights'     :import_weights,
    'weights_list':J2LA(import_weights),
    'nn_info'     :variables_nn,
    }

    T=[]
    invR=None

    prediction=RunNN(title,T,invR,variables,in_weight,model_name,group=None)

if __name__ == '__main__':
    main(usage,options)
