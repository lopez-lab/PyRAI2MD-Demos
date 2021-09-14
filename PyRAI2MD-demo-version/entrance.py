import os
from data_processing import TrainDataInfo
from tools import Read_angle_index

def read_control(keywords,values):
    ## This function read variables from &control
    for i in values:
        if len(i.split()) < 2:
            continue
        key,val=i.split()[0],i.split()[1:]
        key=key.lower()
        if   key == 'title':
            keywords[key] = val[0]
        elif key == 'ml_ncpu':
            keywords[key] = int(val[0])
        elif key == 'qc_ncpu':
            keywords[key] = int(val[0])
        elif key == 'gl_seed':
            keywords[key] = int(val[0])
        elif key == 'jobtype':
       	    keywords[key] = val[0].lower()  # Caution! all jobtype are in lower-case now!
       	elif key == 'qm':
            keywords[key] = val[0].lower()  # Caution! all qm method are in lower-case now!
        elif key == 'abinit':
            keywords[key] = val[0].lower()  # Caution! all abinit method are in lower-case now!
        elif key == 'refine':
            keywords[key] = int(val[0])
        elif key == 'refine_num':
            keywords[key] = int(val[0])
       	elif key == 'refine_start':
            keywords[key] = int(val[0])
       	elif key == 'refine_end':
            keywords[key] = int(val[0])
        elif key == 'maxiter':
            keywords[key] = int(val[0])
        elif key == 'maxsample':
            keywords[key] = int(val[0])
        elif key == 'maxenergy':
            keywords[key] = float(val[0])
        elif key == 'minenergy':
            keywords[key] = float(val[0])
        elif key == 'maxgradient':
            keywords[key] = float(val[0])
        elif key == 'mingradient':
            keywords[key] = float(val[0])
        elif key == 'maxnac':
            keywords[key] = float(val[0])
        elif key == 'minnac':
            keywords[key] = float(val[0])
        elif key == 'neighbor':
            keywords[key] = int(val[0])
        elif key == 'load':
            keywords[key] = int(val[0])
        elif key == 'transfer':
            keywords[key] = int(val[0])
        elif key == 'pop_step':
            keywords[key] = int(val[0])


    return keywords

def read_molcas(keywords,values):
    ## This function read variables from &molcas
    for i in values:
        if len(i.split()) < 2:
            continue
        key,val=i.split()[0],i.split()[1:]
        key=key.lower()
        if   key == 'molcas':
            keywords[key] = val[0]
        elif key == 'molcas_nproc':
            keywords[key] = val[0]
       	elif key == 'molcas_mem':
            keywords[key] = val[0]
       	elif key == 'molcas_print':
            keywords[key] = val[0]
       	elif key == 'molcas_workdir':
            keywords[key] = val[0]
        elif key == 'molcas_calcdir':
            keywords[key] = val[0]
       	elif key == 'omp_num_threads':
            keywords[key] = val[0]
        elif key == 'keep_tmp':
            keywords[key] = int(val[0])
        elif key == 'track_phase':
            keywords[key] = int(val[0])
        elif key == 'basis':
            keywords[key] = int(val[0])
        elif key == 'read_nac':
            keywords[key] = int(val[0])
        elif key == 'use_hpc':
            keywords[key] = int(val[0])

    return keywords

def read_bagel(keywords,values):
    ## This function read variables from &bagel
    for i in values:
        if len(i.split()) < 2:
            continue
        key,val=i.split()[0],i.split()[1:]
        key=key.lower()
        if   key == 'bagel':
            keywords[key] = val[0]
        elif key == 'bagel_nproc':
            keywords[key] = val[0]
        elif key == 'bagel_workdir':
            keywords[key] = val[0]
        elif key == 'bagel_archive':
            keywords[key] = val[0]
        elif key == 'mpi':
            keywords[key] = val[0]
        elif key == 'blas':
            keywords[key] = val[0]
        elif key == 'lapack':
            keywords[key] = val[0]
        elif key == 'boost':
            keywords[key] = val[0]
        elif key == 'mkl':
            keywords[key] = val[0]
        elif key == 'arch':
            keywords[key] = val[0]
        elif key == 'omp_num_threads':
            keywords[key] = val[0]
        elif key == 'keep_tmp':
            keywords[key] = int(val[0])
        elif key == 'read_nac':
            keywords[key] = int(val[0])
        elif key == 'use_mpi':
            keywords[key] = int(val[0])
        elif key == 'use_hpc':
            keywords[key] = int(val[0])

    return keywords

def read_md(keywords,values):
    ## This function read variables from &md
    for i in values:
        if len(i.split()) < 2:
            continue
        key,val=i.split()[0],i.split()[1:]
        key=key.lower()
        if   key == 'initcond':
            keywords[key] = int(val[0])
        elif key == 'excess':
            keywords[key] = float(val[0])
        elif key == 'scale':
            keywords[key] = float(val[0])
        elif key == 'target':
            keywords[key] = float(val[0])
        elif key == 'graddesc':
            keywords[key] = int(val[0])
        elif key == 'reset':
            keywords[key] = int(val[0])
        elif key == 'resetstep':
            keywords[key] = int(val[0])
        elif key == 'nesmb':
            keywords[key] = int(val[0])
       	elif key == 'method':
            keywords[key] = val[0]
       	elif key == 'format':
            keywords[key] = val[0] 
        elif key == 'temp':
            keywords[key] = float(val[0])
        elif key == 'step':
            keywords[key] = int(val[0])
        elif key == 'size':
            keywords[key] = int(val[0])
        elif key == 'ci':
            keywords[key] = int(val[0])
        elif key == 'root':
            keywords[key] = int(val[0])
        elif key == 'sfhp':
            keywords[key] = str(val[0]).lower()
        elif key == 'gap':
            keywords[key] = float(val[0])
        elif key == 'substep':
            keywords[key] = int(val[0])
        elif key == 'deco':
            keywords[key] = val[0]        # Caution! deco must be a string for surfacehopping.py! 
        elif key == 'integrate':
            keywords[key] = int(val[0])
        elif key == 'adjust':
            keywords[key] = int(val[0])
        elif key == 'reflect':
            keywords[key] = int(val[0])
        elif key == 'maxh':
            keywords[key] = int(val[0])
        elif key == 'thermo':
            keywords[key] = int(val[0])
        elif key == 'thermodelay':
            keywords[key] = int(val[0])
        elif key == 'silent':
            keywords[key] = int(val[0])
        elif key == 'verbose':
            keywords[key] = int(val[0])
        elif key == 'direct':
            keywords[key] = int(val[0])
        elif key == 'buffer':
            keywords[key] = int(val[0])
        elif key == 'record':
            keywords[key] = int(val[0])
        elif key == 'restart':
            keywords[key] = int(val[0])
        elif key == 'addstep':
            keywords[key] = int(val[0])
        elif key == 'history':
            keywords[key] = int(val[0])
        elif key == 'ref_e':
            keywords[key] = int(val[0])
        elif key == 'ref_g':
            keywords[key] = int(val[0])
        elif key == 'ref_n':
            keywords[key] = int(val[0])

    return keywords         

def read_gp(keywords,values):
    ## This function read variables from &gp
    for i in values:
        if len(i.split()) < 2:
            continue
        key,val=i.split()[0],i.split()[1:]
        key=key.lower()
        if   key == 'train_data':
            keywords[key] = val[0]
        elif key == 'pred_data':
            keywords[key] = val[0]
        elif key == 'silent':
            keywords[key] = int(val[0])
        elif key == 'modelfile':
            keywords[key] = val[0]
        elif key == 'target' :
            keywords[key] = val[0]
        elif key == 'ratio':
            if   len(val) == 1:
                keywords[key] = [float(val[0]),1-float(val[0])]
            elif len(val) == 2:
                keywords[key] = [float(val[0]),float(val[1])]
        elif key == 'increment':
            keywords[key] = int(val[0])

    keywords['data'],keywords['postdata'],keywords['data_info']=TrainDataInfo(keywords)

    return keywords

def read_nn(keywords,values):
    ## This function read variables from &nn
    for i in values:
        if len(i.split()) < 2:
            continue
        key,val=i.split()[0],i.split()[1:]
        key=key.lower()
        if   key == 'train_data':
            keywords[key] = val[0]
        elif key == 'pred_data':
            keywords[key] = val[0]
        elif key == 'modeldir':
            keywords[key] = val[0]
        elif key == 'silent':
            keywords[key] = int(val[0])
        elif key == 'target' : 
            keywords[key] = val[0]
        elif key == 'ratio':
            if   len(val) == 1:
                keywords[key] = [float(val[0]),1-float(val[0])]
            elif len(val) == 2:
                keywords[key] = [float(val[0]),float(val[1])]
       	elif key == 'increment':
       	    keywords[key] = int(val[0])
        elif key in ['nn_eg_type','nn_nac_type']:
            keywords[key] = int(val[0])
        elif key == 'train_mode':
            keywords[key] = str(val[0]).lower()
        elif key == 'shuffle':
            keywords[key] = {'true':True,'false':False}[val[0].lower()]
        elif key == 'eg_unit':
            keywords[key] = val[0].lower()
        elif key == 'nac_unit':
            keywords[key] = val[0].lower()
        elif key == 'permute_map':
            keywords[key] = str(val[0])
        elif key == 'gpu':
            keywords[key] = int(val[0])

    keywords['data'],keywords['postdata'],keywords['data_info']=TrainDataInfo(keywords)

    return keywords

def read_arch(keywords,values):
    ## This function read variables from &e1,&e2,&g1,&g2,&eg1,&eg2,&nac1,&nac2 
    for i in values:
        if len(i.split()) < 2:
            continue
        key,val=i.split()[0],i.split()[1:]
        key=key.lower()
        if   key == 'depth':
            keywords[key] = int(val[0])
        elif key == 'nn_size':
            keywords[key] = int(val[0])
        elif key == 'activ':
            keywords[key] = str(val[0])
        elif key == 'activ_alpha':
            keywords[key] = float(val[0])
        elif key == 'loss_weights':
            keywords[key] = [float(val[0]),float(val[1])]
        elif key == 'batch_size':
            keywords[key] = int(val[0])
        elif key in ['use_reg_activ','use_reg_weight','use_reg_bias']:
            if str(val[0]).lower() in ['l1','l2','l1_l2']:
                keywords[key] = str(val[0])
            else:
                keywords[key] = None
        elif key == 'reg_l1':
            keywords[key] = float(val[0])
        elif key == 'reg_l2':
            keywords[key] = float(val[0])
       	elif key in ['use_dropout','initialize_weights','phase_less_loss','use_step_callback','use_linear_callback','use_early_callback','use_exp_callback','auto_scaling','scale_x_mean','scale_x_std','scale_energy_mean','scale_energy_std','scale_nac_mean','scale_nac_std']:
            keywords[key] = {'true':True,'false':False,'shifte':'shiftE'}[val[0].lower()]  #'shiftE for auto_scaling'
        elif key in ['val_split','t_val_split','a_val_split']:
            keywords[key] = float(val[0])
        elif key in ['epo','epomin','pre_epo','patience','max_time','epostep','normalization_mode']:
            keywords[key] = int(val[0])
        elif key in ['dropout','delta_loss','factor_lr','learning_rate','learning_rate_start','learning_rate_stop']:
            keywords[key] = float(val[0])
        elif key == 'loss_monitor':
            keywords[key] = str(val[0])
        elif key in ['learning_rate_step']:
            keywords[key] = [float(x) for x in val]
        elif key in ['epoch_step_reduction']:
            keywords[key] = [int(x) for x in val]
        elif key == 'invd_index':
            keywords[key] = Read_angle_index(val,2)
        elif key == 'angle_index':
            keywords[key] = Read_angle_index(val,3)
        elif key == 'dihyd_index':
            keywords[key] = Read_angle_index(val,4)


    return keywords

def ReadInput(input):
    ## This function store all default values for variables
    ## This fucntion read variable from input
    ## This function is expected to be expanded in future as more methods included

    global gl_seed      # global for randome number
    global verbose      # global for MD
    global flr,flrstep  # global for NN

    ## default values
    variables_control={
    'title'       : None,
    'ml_ncpu'     : 1,
    'qc_ncpu'     : 1,
    'gl_seed'     : 1,
    'jobtype'     : None,
    'qm'          :'nn',
    'abinit'      :'molcas',
    'refine'      : 0,
    'refine_num'  : 4,
    'refine_start': 0,
    'refine_end'  : 200,
    'maxiter'     : 1,
    'maxsample'   : 10,
    'maxenergy'   : 0.05,
    'minenergy'   : 0.02,
    'maxgradient' : 0.05,
    'mingradient' : 0.02,
    'maxnac'      : 0.05,
    'minnac'      : 0.02,
    'neighbor'    : 1,
    'load'        : 1,
    'transfer'    : 0,
    'pop_step'    : 200,
    }

    variables_molcas={
    'molcas'         :'/work/lopez/Molcas',
    'molcas_nproc'   :'1',
    'molcas_mem'     :'2000',
    'molcas_print'   :'2',
    'molcas_project' : None,
    'molcas_calcdir' :'%s/tmp_MOLCAS' % (os.getcwd()),
    'molcas_workdir' : None,
    'track_phase'    : 0,
    'basis'          : 2,
    'omp_num_threads':'1',
    'read_nac'       : 1,
    'use_hpc'        : 0,
    'ci'             : 0,     # Caution! This value will be updated by variables_md['ci']. Not allow user to set.
    'previous_civec' : None,  # Caution! This value will be set when ci vector read from molcas. Not allow user to set.
    'previous_movec' : None,  # Caution! This value will be set when mo vector read from molcas. Not allow user to set.
    'group'          : None,  # Caution! This value will be set when run multiple molcas. Not allow user to set.
    'keep_tmp'       : 1,
    'verbose'        : 0,
    }

    variables_bagel={
    'bagel'          :'/work/lopez/Bagel-mvapich',
    'bagel_nproc'    : 1,
    'bagel_project'  : None,
    'bagel_workdir'  :'%s/tmp_BAGEL' % (os.getcwd()),
    'bagel_archive'  :'default',
    'mpi'            :'/work/lopez/mvapich2-2.3.4',
    'blas'           :'/work/lopez/BLAS',
    'lapack'         :'/work/lopez/BLAS',
    'boost'          :'/work/lopez/Boost',
    'mkl'            :'/work/lopez/intel/mkl/bin/mklvars.sh',
    'arch'           :'intel64',
    'omp_num_threads':'1',
    'read_nac'       : 0,
    'use_mpi'        : 0,
    'use_hpc'  	     : 0,
    'ci'             : 0,     # Caution! This value will be updated by variables_md['ci']. Not allow user to set.
    'previous_civec' : None,  # Caution! BAGEL does not use this value
    'previous_movec' : None,  # Caution! BAGEL does not use this value
    'group'          : None,  # Caution! This value will be set when run multiple BAGEL. Not allow user to set.
    'keep_tmp'       : 1,
    'verbose'        : 0,
    }

    variables_md={
    'gl_seed'     : 1,     # Caution! This value will be updated by variables_control['gl_seed']. Not allow user to set.
    'initcond'    : 1,
    'excess'      : 0,
    'scale'       : 1,
    'target'      : 0,
    'graddesc'    : 0,
    'reset'       : 1,
    'resetstep'   : 0,
    'nesmb'       : 20,
    'method'      :'wigner',
    'format'      :'molden',
    'temp'        : 300,
    'step'        : 10,
    'size'        : 40,
    'ci'          : 2,
    'root'        : 2,
    'sfhp'        :'nosh',
    'gap'         : 0.5,
    'substep'     : 0,
    'integrate'   : 0,
    'deco'        : '0.1',
    'adjust'      : 1,
    'reflect'     : 1,
    'maxh'        : 10,
    'thermo'      : -1,
    'thermodelay' : 200,
    'silent'      : 1,
    'verbose'     : 0,
    'direct'      : 2000,
    'buffer'      : 500,
    'record'      : 1,
    'restart'     : 0,
    'addstep'     : 0,
    'history'     : 100,
    'group'       : None,    # Caution! This value will be set when run multiple md. Not allow user to set.
    'ref_e'       : 0,
    'ref_g'       : 0,
    'ref_n'       : 0, 
   }

    variables_gp={
    'train_data' : None,
    'pred_data'  : None,
    'silent'     : 0,
    'target'     :'all',
    'ratio'      :[0.9, 0.1],
    'increment'  : 0,
    'model'      : None,
    'modelfile'  : None,  # Caution! This value will be updated by read_gp. Not allow user to set.
    'ml_seed'    : 1,     # Caution! This value will be updated by variables_control['gl_seed']. Not allow user to set.
    'data'	 : None,  # Caution! This value will be updated by TrainDataInfo. Not allow user to set.
    'postdata'   : None,  # Caution! This value will be updated by TrainDataInfo. Not allow user to set.
    'data_info'  : None,  # Caution! This value will be updated by TrainDataInfo. Not allow user to set.
    }

    variables_nn={
    'train_mode' :'training',
    'train_data' : None,
    'pred_data'  : None,
    'modeldir'   : None,
    'ml_seed'    : 1,     # Caution! This value will be updated by variables_control['gl_seed']. Not allow user to set.
    'silent'     : 1,
    'target'     :'eg1',
    'ratio'      :[0.9, 0.1],
    'increment'  : 0,
    'data'       : None,  # Caution! This value will be updated by TrainDataInfo. Not allow user to set.
    'postdata'   : None,  # Caution! This value will be updated by TrainDataInfo. Not allow user to set.
    'data_info'  : None,  # Caution! This value will be updated by TrainDataInfo. Not allow user to set.
    'nn_eg_type' : 1,
    'nn_nac_type': 1,
    'shuffle'    : False,
    'eg_unit'    :'si',
    'nac_unit'   :'au',
    'eg'         : None,  # Caution! This value will be updated later. Not allow user to set.
    'nac'      	 : None,  # Caution! This value will be updated later. Not allow user to set.
    'eg2'        : None,  # Caution! This value will be updated later. Not allow user to set.
    'nac2'       : None,  # Caution! This value will be updated later. Not allow user to set.
    'permute_map':'No',
    'gpu'        : 0,
    }

    variables_eg={
    'model_type'            :'mlp_eg',
    'invd_index'            :[],
    'angle_index'           :[],
    'dihyd_index'           :[],
    'depth'                 : 4,
    'nn_size'               : 100,
    'activ'                 :'leaky_softplus',
    'activ_alpha'           : 0.03,
    'loss_weights'          :[1, 1],
    'use_dropout'           : False,
    'dropout'               : 0.005,
    'use_reg_activ'    	    : None,
    'use_reg_weight'        : None,
    'use_reg_bias'          : None,
    'reg_l1'                : 1e-5,
    'reg_l2'                : 1e-5,
    'use_step_callback'     : True,
    'use_linear_callback'   : False,
    'use_early_callback'    : False,
    'use_exp_callback'      : False,
    'scale_x_mean'          : False,
    'scale_x_std'           : False,
    'scale_y_mean'          : True,
    'scale_y_std'           : True,
    'normalization_mode'    : 1,
    'learning_rate'         : 1e-3,
    'initialize_weights'    : True,
    'val_disjoint'          : True,
    'val_split'             : 0.1,
    'epo'                   : 2000,
    'epomin'                : 1000,
    'patience'              : 300,
    'max_time'              : 300,
    'batch_size'            : 64,
    'delta_loss'            : 1e-5,
    'loss_monitor'          :'val_loss',
    'factor_lr'             : 0.1,
    'epostep'               : 10,
    'learning_rate_start'   : 1e-3,
    'learning_rate_stop'    : 1e-6,
    'learning_rate_step'    : [1e-3, 1e-4, 1e-5, 1e-6],
    'epoch_step_reduction'  : [500, 500, 500, 500],
    }

    variables_nac={
    'model_type'            :'mlp_nac2',
    'invd_index'       	    :[],
    'angle_index'           :[],
    'dihyd_index'           :[],
    'depth'                 : 4,
    'nn_size'               : 100,
    'activ'                 :'leaky_softplus',
    'activ_alpha'           : 0.03,
    'use_dropout'      	    : False,
    'dropout'  	       	    : 0.005,
    'use_reg_activ'         : None,
    'use_reg_weight'        : None,
    'use_reg_bias'          : None,
    'reg_l1'                : 1e-5,
    'reg_l2'                : 1e-5,
    'use_step_callback'	    : True,
    'use_linear_callback'   : False,
    'use_early_callback'    : False,
    'use_exp_callback' 	    : False,
    'scale_x_mean'          : False,
    'scale_x_std'           : False,
    'scale_y_mean'          : True,
    'scale_y_std'           : True,
    'normalization_mode'    : 1,
    'learning_rate'         : 1e-3,
    'phase_less_loss'       : False,
    'initialize_weights'    : True,
    'val_disjoint'          : True,
    'val_split'             : 0.1,
    'epo'                   : 2000,
    'epomin'                : 1000,
    'pre_epo'               : 100,
    'patience'              : 300,
    'max_time'              : 300,
    'batch_size'            : 64,
    'delta_loss'            : 1e-5,
    'loss_monitor'          :'val_loss',
    'factor_lr'             : 0.1,
    'epostep'               : 10,
    'learning_rate_start'   : 1e-3,
    'learning_rate_stop'    : 1e-6,
    'learning_rate_step'    : [1e-3, 1e-4, 1e-5, 1e-6],
    'epoch_step_reduction'  : [500, 500, 500, 500],
    }

    ## More default will be added below

    ## ready to read input
    variables_input={
    'control': variables_control,
    'molcas' : variables_molcas,
    'bagel'  : variables_bagel,
    'md'     : variables_md,
    'gp'     : variables_gp,
    'nn'     : variables_nn,
    'eg'     : variables_eg.copy(),
    'nac'    : variables_nac.copy(),
    'eg2'    : variables_eg.copy(),
    'nac2'   : variables_nac.copy(),
    }

    variables_readfunc={
    'control': read_control,
    'molcas' : read_molcas,
    'bagel'  : read_bagel,
    'md'     : read_md,
    'gp'     : read_gp,
    'nn'     : read_nn,
    'eg'     : read_arch,
    'nac'    : read_arch,
    'eg2'    : read_arch,
    'nac2'   : read_arch,
    }

    ## read input variable:
    for i in input:
        i=i.splitlines()
        if len(i) == 0:
            continue
        variable_name=i[0].lower()
        variables_input[variable_name]=variables_readfunc[variable_name](variables_input[variable_name],i)

    ## assemble variables
    variables_all={
    'control': variables_input['control'],
    'molcas' : variables_input['molcas'],
    'bagel'  : variables_input['bagel'],
    'md'     : variables_input['md'],
    'gp'     : variables_input['gp'],
    'nn'     : variables_input['nn'],
    }

    ## update variables_nn
    variables_all['nn']['eg']    = variables_input['eg']
    variables_all['nn']['nac']   = variables_input['nac']
    variables_all['nn']['eg2']   = variables_input['eg2']
    variables_all['nn']['nac2']  = variables_input['nac2']

    variables_all['md']['gl_seed']            = variables_all['control']['gl_seed']
    variables_all['gp']['ml_seed']            = variables_all['control']['gl_seed']
    variables_all['nn']['ml_seed']            = variables_all['control']['gl_seed']
    variables_all['molcas']['molcas_project'] = variables_all['control']['title']
    variables_all['molcas']['ci']             = variables_all['md']['ci']
    variables_all['molcas']['verbose']        = variables_all['md']['verbose']
    variables_all['bagel']['bagel_project']   = variables_all['control']['title']
    variables_all['bagel']['ci']              = variables_all['md']['ci']
    variables_all['bagel']['verbose']         = variables_all['md']['verbose']

    return variables_all

def StartInfo(variables_all):
    ##  This funtion print start information 

    variables_control = variables_all['control']
    variables_molcas  = variables_all['molcas']
    variables_bagel   = variables_all['bagel']
    variables_md      = variables_all['md']
    variables_gp      = variables_all['gp']
    variables_nn      = variables_all['nn']
    variables_eg      = variables_nn['eg']
    variables_nac     = variables_nn['nac']
    variables_eg2     = variables_nn['eg2']
    variables_nac2    = variables_nn['nac2']

    ## unpack control variables
    title             = variables_all['control']['title']
    jobtype           = variables_all['control']['jobtype']
    qm                = variables_all['control']['qm']
    abinit            = variables_all['control']['abinit']

    control_info="""
  &control
-------------------------------------------------------
  Title:                      %-10s
  NCPU for ML:                %-10s
  NCPU for QC:                %-10s
  Seed:                       %-10s
  Job: 	                      %-10s
  QM:          	       	      %-10s
-------------------------------------------------------
""" % (variables_control['title'],   variables_control['ml_ncpu'], variables_control['qc_ncpu'],\
       variables_control['gl_seed'], variables_control['jobtype'], variables_control['qm'])


    adaptive_info="""
  &adaptive sampling method
-------------------------------------------------------
  Ab initio:                  %-10s
  Load model:                 %-10s
  Transfer learning:          %-10s
  Maxiter:                    %-10s
  Refine crossing:            %-10s
  Refine points/range: 	      %-10s %-10s %-10s
  Max/Min energy:             %-10s %-10s
  Max/Min gradient:           %-10s %-10s
  Max/Min nac:                %-10s %-10s
-------------------------------------------------------
""" % (variables_control['abinit'],       variables_control['load'],\
       variables_control['transfer'],     variables_control['maxiter'],\
       variables_control['refine'],       variables_control['refine_num'],\
       variables_control['refine_start'], variables_control['refine_end'],\
       variables_control['maxenergy'],    variables_control['minenergy'],\
       variables_control['maxgradient'],  variables_control['mingradient'],\
       variables_control['maxnac'],       variables_control['minnac'])

    md_info="""
  &initial condition
-------------------------------------------------------
  Generate initial condition: %-10s
  Number:                     %-10s
  Method:                     %-10s 
  Format:                     %-10s
-------------------------------------------------------
 
  &md
-------------------------------------------------------
  CI dimension:               %-10s
  Initial root:               %-10s
  Temperature (K):            %-10s
  Step:                       %-10s
  Dt (au):                    %-10s
  Surface hopping:            %-10s
  Substep:                    %-10s
  Integrate probability       %-10s
  Decoherance:                %-10s
  Adjust velocity:            %-10s
  Reflect velocity:           %-10s
  Maxhop:                     %-10s
  Thermostat:                 %-10s
  Thermostat delay:           %-10s
  Print level:                %-10s
  Direct output:              %-10s
  Buffer output:              %-10s
  Record MD history:          %-10s
  Restart function:           %-10s
  Additional steps:           %-10s
  History:                    %-10s
-------------------------------------------------------

  &md velocity control
-------------------------------------------------------
  Excess kinetic energy       %-10s
  Scale kinetic energy        %-10s
  Target kinetic energy       %-10s
  Gradient descent path       %-10s
  Reset velocity:             %-10s
  Reset step:                 %-10s
-------------------------------------------------------

""" % (variables_md['initcond'], variables_md['nesmb'],        variables_md['method'],  variables_md['format'], \
       variables_md['ci'],       variables_md['root'],         variables_md['temp'],    variables_md['step'],   \
       variables_md['size'],     variables_md['sfhp'],         variables_md['substep'], variables_md['integrate'], \
       variables_md['deco'],     variables_md['adjust'],       variables_md['reflect'], variables_md['maxh'],\
       variables_md['thermo'],   variables_md['thermodelay'],  variables_md['verbose'], variables_md['direct'],\
       variables_md['buffer'],   variables_md['record'],       variables_md['restart'], variables_md['addstep'],\
       variables_md['history'],\
       variables_md['excess'],   variables_md['scale'],        variables_md['target'],\
       variables_md['graddesc'], variables_md['reset'],        variables_md['resetstep'])

    hybrid_info="""
  &hybrid namd
-------------------------------------------------------
  Mix Energy                  %-10s
  Mix Gradient                %-10s
  Mix NAC                     %-10s
-------------------------------------------------------
""" % (variables_md['ref_e'],variables_md['ref_g'],variables_md['ref_n'])

    gp_info="""
%s

  &gp
-------------------------------------------------------
  Train data:                 %-10s
  Predition data:             %-10s
  Silent mode:                %-10s
  Model file:                 %-10s
-------------------------------------------------------
""" % (variables_gp['data_info'], variables_gp['train_data'], variables_gp['pred_data'], variables_gp['silent'], variables_gp['model'])
 
    nn_info="""
%s

  &nn
-------------------------------------------------------
  Train data:                 %-10s
  Predition data:             %-10s
  Train mode:                 %-10s
  Silent mode:                %-10s
  NN EG type:                 %-10s
  NN NAC type:                %-10s
  Shuffle data:               %-10s
  EG unit:                    %-10s
  NAC unit:                   %-10s
  Data permutation            %-10s
-------------------------------------------------------

  &hyperparameters            Energy+Gradient    Nonadiabatic couplings Energy+Gradient(2) Nonadiabatic couplings(2)
-----------------------------------------------------------------------
  InvD features:              %-20s %-20s %-20s %-20s 
  Angle features:             %-20s %-20s %-20s %-20s
  Dihedral features:          %-20s %-20s %-20s %-20s
  Activation:                 %-20s %-20s %-20s %-20s
  Activation alpha:           %-20s %-20s %-20s %-20s
  Layers:      	              %-20s %-20s %-20s %-20s
  Neurons/layer:              %-20s %-20s %-20s %-20s
  Dropout:                    %-20s %-20s %-20s %-20s
  Dropout ratio:              %-20s %-20s %-20s %-20s
  Regularization activation:  %-20s %-20s %-20s %-20s
  Regularization weight:      %-20s %-20s %-20s %-20s
  Regularization bias:        %-20s %-20s %-20s %-20s
  L1:                         %-20s %-20s %-20s %-20s
  L2:         	              %-20s %-20s %-20s %-20s
  Phase-less loss:            %-20s %-20s %-20s %-20s
  Initialize weight:          %-20s %-20s %-20s %-20s
  Validation disjoint:        %-20s %-20s %-20s %-20s
  Validation split:           %-20s %-20s %-20s %-20s
  Epoch:                      %-20s %-20s %-20s %-20s
  Epoch_pre:                  %-20s %-20s %-20s %-20s
  Epoch_min:                  %-20s %-20s %-20s %-20s
  Patience:                   %-20s %-20s %-20s %-20s
  Max time:                   %-20s %-20s %-20s %-20s
  Epoch step:                 %-20s %-20s %-20s %-20s
  Batch:                      %-20s %-20s %-20s %-20s
  Delta loss:                 %-20s %-20s %-20s %-20s
  Shift_X:     	       	      %-20s %-20s %-20s %-20s
  Scale_X:                    %-20s %-20s %-20s %-20s
  Shift_Y:                    %-20s %-20s %-20s %-20s
  Scale_Y:                    %-20s %-20s %-20s %-20s
  Feature normalization:      %-20s %-20s %-20s %-20s
-----------------------------------------------------------------------
""" % (variables_nn['data_info'],          variables_nn['train_data'],          variables_nn['pred_data'],           variables_nn['train_mode'],\
       variables_nn['silent'],             variables_nn['nn_eg_type'],          variables_nn['nn_nac_type'],         variables_nn['shuffle'],\
       variables_nn['eg_unit'],            variables_nn['nac_unit'],            variables_nn['permute_map'],\
       len(variables_eg['invd_index']),    len(variables_nac['invd_index']),    len(variables_eg['invd_index']),     len(variables_nac2['invd_index']),\
       len(variables_eg['angle_index']),   len(variables_nac['angle_index']),   len(variables_eg['angle_index']),    len(variables_nac2['angle_index']),\
       len(variables_eg['dihyd_index']),   len(variables_nac['dihyd_index']),  	len(variables_eg['dihyd_index']),    len(variables_nac2['dihyd_index']),\
       variables_eg['activ'],              variables_nac['activ'],              variables_eg2['activ'],              variables_nac2['activ'],\
       variables_eg['activ_alpha'],        variables_nac['activ_alpha'],        variables_eg2['activ_alpha'],        variables_nac2['activ_alpha'],\
       variables_eg['depth'],              variables_nac['depth'],              variables_eg2['depth'],              variables_nac2['depth'],\
       variables_eg['nn_size'],            variables_nac['nn_size'],            variables_eg2['nn_size'],            variables_nac2['nn_size'],\
       variables_eg['use_dropout'],        variables_nac['use_dropout'],        variables_eg2['use_dropout'],        variables_nac2['use_dropout'],\
       variables_eg['dropout'],            variables_nac['dropout'],            variables_eg2['dropout'],            variables_nac2['dropout'],\
       variables_eg['use_reg_activ'],      variables_nac['use_reg_activ'],      variables_eg2['use_reg_activ'],      variables_nac2['use_reg_activ'],\
       variables_eg['use_reg_weight'],     variables_nac['use_reg_weight'],     variables_eg2['use_reg_weight'],     variables_nac2['use_reg_weight'],\
       variables_eg['use_reg_bias'],       variables_nac['use_reg_bias'],       variables_eg2['use_reg_bias'],       variables_nac2['use_reg_bias'],\
       variables_eg['reg_l1'],             variables_nac['reg_l1'],             variables_eg2['reg_l1'],             variables_nac2['reg_l1'],\
       variables_eg['reg_l2'],             variables_nac['reg_l2'],             variables_eg2['reg_l2'],             variables_nac2['reg_l2'],\
       'N/A',                              variables_nac['phase_less_loss'],    'N/A',                               variables_nac2['phase_less_loss'],\
       variables_eg['initialize_weights'], variables_nac['initialize_weights'], variables_eg2['initialize_weights'], variables_nac2['initialize_weights'],\
       variables_eg['val_disjoint'],       variables_nac['val_disjoint'],       variables_eg2['val_disjoint'],       variables_nac2['val_disjoint'],\
       variables_eg['val_split'],          variables_nac['val_split'],          variables_eg2['val_split'],          variables_nac2['val_split'],\
       variables_eg['epo'],                variables_nac['epo'],                variables_eg2['epo'],                variables_nac2['epo'],\
       'N/A',                              variables_nac['pre_epo'],            'N/A',                               variables_nac2['pre_epo'],\
       variables_eg['epomin'],             variables_nac['epomin'],             variables_eg2['epomin'],             variables_nac2['epomin'], \
       variables_eg['patience'],           variables_nac['patience'],           variables_eg2['patience'],           variables_nac2['patience'],\
       variables_eg['max_time'],           variables_nac['max_time'],           variables_eg2['max_time'],           variables_nac2['max_time'],\
       variables_eg['epostep'],            variables_nac['epostep'],            variables_eg2['epostep'],            variables_nac2['epostep'],\
       variables_eg['batch_size'],         variables_nac['batch_size'],         variables_eg2['batch_size'],         variables_nac2['batch_size'],\
       variables_eg['delta_loss'],         variables_nac['delta_loss'],         variables_eg2['delta_loss'],         variables_nac2['delta_loss'],\
       variables_eg['scale_x_mean'],       variables_nac['scale_x_mean'],       variables_eg2['scale_x_mean'],       variables_nac2['scale_x_mean'],\
       variables_eg['scale_x_std'],        variables_nac['scale_x_std'],        variables_eg2['scale_x_std'],        variables_nac2['scale_x_std'],\
       variables_eg['scale_y_mean'],       variables_nac['scale_y_mean'],       variables_eg2['scale_y_mean'],       variables_nac2['scale_y_mean'],\
       variables_eg['scale_y_std'],        variables_nac['scale_y_std'],        variables_eg2['scale_y_std'],        variables_nac2['scale_y_std'],\
       variables_eg['normalization_mode'], variables_nac['normalization_mode'], variables_eg2['normalization_mode'], variables_nac2['normalization_mode'])
    molcas_info="""
  &molcas
-------------------------------------------------------
  Molcas:                   %-10s
  Molcas_nproc:             %-10s
  Molcas_mem:               %-10s
  Molcas_print:      	    %-10s
  Molcas_project:      	    %-10s
  Molcas_workdir:      	    %-10s
  Molcas_calcdir:           %-10s
  Omp_num_threads:          %-10s
  State:                    %-10s
  Keep tmp_molcas:          %-10s
  Track phase:              %-10s
  Read NAC:                 %-10s
  Submit jobs:              %-10s
-------------------------------------------------------
""" % (variables_molcas['molcas'],          variables_molcas['molcas_nproc'],    variables_molcas['molcas_mem'],     \
       variables_molcas['molcas_print'],    variables_molcas['molcas_project'],  variables_molcas['molcas_workdir'], \
       variables_molcas['molcas_calcdir'],  variables_molcas['omp_num_threads'], variables_molcas['ci'],\
       variables_molcas['keep_tmp'],        variables_molcas['track_phase'],     variables_molcas['read_nac'],\
       variables_molcas['use_hpc'])

    bagel_info="""
  &bagel
-------------------------------------------------------
  BAGEL:                    %-10s
  BAGEL_nproc:              %-10s
  BAGEL_project:            %-10s
  BAGEL_workdir:            %-10s
  BAGEL_archive:            %-10s
  MPI:                      %-10s
  BLAS:                     %-10s
  LAPACK:                   %-10s
  BOOST:                    %-10s
  MKL:                      %-10s
  Architecture:             %-10s
  Omp_num_threads:          %-10s
  State:                    %-10s
  Keep tmp_bagel:           %-10s
  Read NAC:                 %-10s
  Submit jobs:              %-10s
-------------------------------------------------------
""" % (variables_bagel['bagel'],         variables_bagel['bagel_nproc'],     variables_bagel['bagel_project'],\
       variables_bagel['bagel_workdir'], variables_bagel['bagel_archive'],\
       variables_bagel['mpi'],           variables_bagel['blas'],\
       variables_bagel['lapack'],        variables_bagel['boost'],           variables_bagel['mkl'],\
       variables_bagel['arch'],          variables_bagel['omp_num_threads'], variables_bagel['ci'],\
       variables_bagel['keep_tmp'],      variables_bagel['read_nac'],        variables_bagel['use_hpc'])

    info_method={
    'gp'    :   gp_info,
    'nn'    :   nn_info,
    'molcas':   molcas_info,
    'bagel' :   bagel_info
    }

    info_abinit={
    'molcas':   molcas_info,
    'bagel' :   bagel_info
    }

    qm     = variables_control['qm']
    abinit = variables_control['abinit']
    info_jobtype={
    'md':         control_info+              md_info+info_method[qm],
    'hybrid':     control_info+              md_info+info_method[qm]+info_abinit[abinit]+hybrid_info,
    'adaptive':   control_info+adaptive_info+md_info+info_method[qm]+info_abinit[abinit],
    'train':      control_info+                      info_method[qm],
    'prediction': control_info+                      info_method[qm],
    'search':     control_info+                      info_method[qm],
    }

    log_info=info_jobtype[jobtype]

    return log_info

