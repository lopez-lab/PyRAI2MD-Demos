## Neural Networks interface for PyRAIMD
## Jingbai Li Jul 0 2020

import time,datetime,json
import numpy as np
from tools import PermuteMap
from pyNNsMD.nn_pes import NeuralNetPes
from pyNNsMD.nn_pes_src.device import set_gpu

class DNN:
    ## This is the interface to GP

    def __init__(self,variables_all,id=None):
        ## data      : dict
        ##             All data from traning data
        ## pred_data : str
        ##             Filename for test set
        ## hyp_eg/nac: dict
        ##           : Hyperparameters for NNs
        ## x         : np.array
        ##             Inverse distance in shape of (batch,(atom*atom-1)/2)
        ## y_dict    : dict
        ##             Dictionary of y values for each model. Energy in Bohr, Gradients in Hatree/Bohr. Nac are unchanged.

        ## unpack variables
        set_gpu([]) #No GPU for prediction
        title                      = variables_all['control']['title']
        variables                  = variables_all['nn']
        modeldir                   = variables['modeldir']
        data                       = variables['postdata']
        nn_eg_type                 = variables['nn_eg_type']
        nn_nac_type                = variables['nn_nac_type']
        hyp_eg                     = variables['eg'].copy()
        hyp_nac                    = variables['nac'].copy()
        hyp_eg2                    = variables['eg2'].copy()
        hyp_nac2                   = variables['nac2'].copy()
        seed                       = variables['ml_seed']
        permute                    = variables['permute_map']
        gpu                        = variables['gpu']

        ## setup l1 l2 dict
        for model_dict in [hyp_eg,hyp_nac,hyp_eg2,hyp_nac2]:
            for penalty in ['use_reg_activ','use_reg_weight','use_reg_bias']:
                penalty_key='%s_dict' % (penalty)
                if   model_dict[penalty] == 'l1':
                    model_dict[penalty_key] = {
                                              'class_name' : 'l1',
                                              'config'     : {
                                                             'l1':model_dict['reg_l1'],
                                                             },
                                              }
                elif model_dict[penalty] == 'l2':
                    model_dict[penalty_key] = {
                                              'class_name' : 'l2',
                                              'config'     : {
                                                             'l2':model_dict['reg_l2'],
                                                             },
                                              }
                elif model_dict[penalty] == 'l1_l2':
                    model_dict[penalty_key] = {
                                              'class_name' : 'l1_l2',
                                              'config'     : {
                                                             'l1':model_dict['reg_l1'],
                                                             'l2':model_dict['reg_l2'],
                                                             },
                                              }
                else:
                    model_dict[penalty_key] = None

        ## setup unit scheme
        if variables['eg_unit'] == 'si':
            hyp_eg['unit']   = ['eV','eV/A']
       	    hyp_eg2['unit']  = ['eV','eV/A']
        else:
            hyp_eg['unit']   = ['Eh','Eh/Bohr']
       	    hyp_eg2['unit']  = ['Eh','Eh/Bohr']
        if variables['nac_unit'] == 'si':
       	    hyp_nac['unit']  = 'eV/A'
            hyp_nac2['unit'] = 'eV/A'
        elif variables['nac_unit'] == 'au':
            hyp_nac['unit']  = 'Eh/A'
            hyp_nac2['unit'] = 'Eh/A'
        elif variables['nac_unit'] == 'eha':
       	    hyp_nac['unit']  = 'Eh/Bohr'
       	    hyp_nac2['unit'] = 'Eh/Bohr'

        ## setup hypers
        hyp_dict_eg  ={
                      'general'    :{
                                    'model_type'            : hyp_eg['model_type'],
                                    },
                      'model'      :{
                                    'atoms'                 : data['natom'],
                                    'states'                : data['nstate'], 
                                    'nn_size'               : hyp_eg['nn_size'],
                                    'depth'                 : hyp_eg['depth'],
                                    'activ'                 : {
                                                               'class_name' : hyp_eg['activ'],
                                                               'config'     : {
                                                                               'alpha' : hyp_eg['activ_alpha'],
                                                                              },
                                                              },
                                    'use_dropout'           : hyp_eg['use_dropout'],
                                    'dropout'               : hyp_eg['dropout'],
                                    'use_reg_activ'         : hyp_eg['use_reg_activ_dict'],
                                    'use_reg_weight'        : hyp_eg['use_reg_weight_dict'],
                                    'use_reg_bias'          : hyp_eg['use_reg_bias_dict'],
                                    'invd_index'            : hyp_eg['invd_index'],
                                    'angle_index'           : hyp_eg['angle_index'],
                                    'dihyd_index'           : hyp_eg['dihyd_index'],
                                    },
       	       	      'training'   :{
                                    'auto_scaling'          : {
                                                               'x_mean'               :hyp_eg['scale_x_mean'],
                                                               'x_std'                :hyp_eg['scale_x_std'],
                                                               'energy_mean'          :hyp_eg['scale_y_mean'],
                                                               'energy_std'           :hyp_eg['scale_y_std'],
                                                              },

                                    'normalization_mode'    : hyp_eg['normalization_mode'],
                                    'loss_weights'          : hyp_eg['loss_weights'],
                                    'learning_rate'         : hyp_eg['learning_rate'],
                                    'initialize_weights'    : hyp_eg['initialize_weights'],
                                    'val_disjoint'          : hyp_eg['val_disjoint'],
                                    'val_split'             : hyp_eg['val_split'],
                                    'epo'                   : hyp_eg['epo'],
                                    'batch_size'            : hyp_eg['batch_size'],
                                    'epostep'               : hyp_eg['epostep'],
                                    'step_callback'         : {
                                                               'use'                  : hyp_eg['use_step_callback'],
                                                               'epoch_step_reduction' : hyp_eg['epoch_step_reduction'],
                                                               'learning_rate_step'   : hyp_eg['learning_rate_step'],
                                                              },
                                    'linear_callback'       : {
                                                               'use'                  : hyp_eg['use_linear_callback'],
                                                               'learning_rate_start'  : hyp_eg['learning_rate_start'],
                                                               'learning_rate_stop'   : hyp_eg['learning_rate_stop' ],
                                                               'epomin'               : hyp_eg['epomin'],
                                                              },
       	       	       	       	    'early_callback'        : {
                                                               'use'                  : hyp_eg['use_early_callback'],
                                                               'epomin'               : hyp_eg['epomin'],
                                                               'patience'             : hyp_eg['patience'],
                                                               'max_time'             : hyp_eg['max_time'],
                                                               'delta_loss'           : hyp_eg['delta_loss'],
                                                               'loss_monitor'         : hyp_eg['loss_monitor'],
                                                               'factor_lr'            : hyp_eg['factor_lr'],
                                                               'learning_rate_start'  : hyp_eg['learning_rate_start'],
                                                               'learning_rate_stop'   : hyp_eg['learning_rate_stop' ],
                                                              },
       	       	       	       	    'exp_callback'          : {
                                                               'use'                  : hyp_eg['use_exp_callback'],
                                                               'epomin'               : hyp_eg['epomin'],
                                                               'factor_lr'            : hyp_eg['factor_lr'],
                                                              },
       	       	                    },
       	       	      'plots'      :{
                                    'unit_energy'           : hyp_eg['unit'][0],
       	       	       	       	    'unit_gradient'         : hyp_eg['unit'][1],
       	       	                    },
                      }

        hyp_dict_nac ={
                      'general'    :{
                                    'model_type'            : hyp_nac['model_type'],
                                    },
                      'model'      :{
                                    'atoms'                 : data['natom'],
                                    'states'                : data['nstate'], 
                                    'nn_size'               : hyp_nac['nn_size'],
                                    'depth'                 : hyp_nac['depth'],
                                    'activ'                 : {
                                                               'class_name' : hyp_nac['activ'],
                                                               'config'     : {
                                                                               'alpha' : hyp_nac['activ_alpha'],
                                                                              },
                                                              },
                                    'use_dropout'           : hyp_nac['use_dropout'],
                                    'dropout'               : hyp_nac['dropout'],
                                    'use_reg_activ'         : hyp_nac['use_reg_activ_dict'],
                                    'use_reg_weight'        : hyp_nac['use_reg_weight_dict'], 
                                    'use_reg_bias'          : hyp_nac['use_reg_bias_dict'],
                                    'invd_index'            : hyp_nac['invd_index'],
                                    'angle_index'           : hyp_nac['angle_index'],
                                    'dihyd_index'           : hyp_nac['dihyd_index'],
                                    },
       	       	      'training'   :{
                                    'auto_scaling'          : {
                                                               'x_mean'               :hyp_nac['scale_x_mean'],
                                                               'x_std'                :hyp_nac['scale_x_std'],
                                                               'nac_mean'             :hyp_nac['scale_y_mean'],
                                                               'nac_std'              :hyp_nac['scale_y_std'],
                                                              },
                                    'normalization_mode'    : hyp_nac['normalization_mode'],
                                    'learning_rate'         : hyp_nac['learning_rate'],
                                    'phase_less_loss'       : hyp_nac['phase_less_loss'],
                                    'initialize_weights'    : hyp_nac['initialize_weights'],
                                    'val_disjoint'          : hyp_nac['val_disjoint'],
                                    'val_split'             : hyp_nac['val_split'],
                                    'epo'                   : hyp_nac['epo'],
                                    'pre_epo'               : hyp_nac['pre_epo'],
                                    'batch_size'            : hyp_nac['batch_size'],
                                    'epostep'               : hyp_nac['epostep'],
                                    'step_callback'         : {
                                                               'use'                  : hyp_nac['use_step_callback'],
                                                               'epoch_step_reduction' : hyp_nac['epoch_step_reduction'],
                                                               'learning_rate_step'   : hyp_nac['learning_rate_step'],
                                                              },
                                    'linear_callback'       : {
                                                               'use'                  : hyp_nac['use_linear_callback'],
                                                               'learning_rate_start'  : hyp_nac['learning_rate_start'],
                                                               'learning_rate_stop'   : hyp_nac['learning_rate_stop' ],
                                                               'epomin'               : hyp_nac['epomin'],
                                                              },
       	       	       	       	    'early_callback'        : {
                                                               'use'                  : hyp_nac['use_early_callback'],
                                                               'epomin'               : hyp_nac['epomin'],
                                                               'patience'             : hyp_nac['patience'],
                                                               'max_time'             : hyp_nac['max_time'],
                                                               'delta_loss'           : hyp_nac['delta_loss'],
                                                               'loss_monitor'         : hyp_nac['loss_monitor'],
                                                               'factor_lr'            : hyp_nac['factor_lr'],
                                                               'learning_rate_start'  : hyp_nac['learning_rate_start'],
                                                               'learning_rate_stop'   : hyp_nac['learning_rate_stop' ],
                                                              },
       	       	       	       	    'exp_callback'          : {
                                                               'use'                  : hyp_nac['use_exp_callback'],
                                                               'epomin'               : hyp_nac['epomin'],
                                                               'factor_lr'            : hyp_nac['factor_lr'],
                                                              },
       	       	                    },
       	       	      'plots'      :{
                                    'unit_nac'              : hyp_nac['unit'],
       	       	                    },
                      }

        hyp_dict_eg2 ={
                      'general'    :{
                                    'model_type'            : hyp_eg2['model_type'],
                                    },
                      'model'      :{
                                    'atoms'                 : data['natom'],
                                    'states'                : data['nstate'], 
                                    'nn_size'               : hyp_eg2['nn_size'],
                                    'depth'                 : hyp_eg2['depth'],
                                    'activ'                 : {
                                                               'class_name' : hyp_eg2['activ'],
                                                               'config'     : {
                                                                               'alpha' : hyp_eg2['activ_alpha'],
                                                                              },
                                                              },
                                    'use_dropout'           : hyp_eg2['use_dropout'],
                                    'dropout'               : hyp_eg2['dropout'],
                                    'use_reg_activ'         : hyp_eg2['use_reg_activ_dict'],
                                    'use_reg_weight'        : hyp_eg2['use_reg_weight_dict'], 
                                    'use_reg_bias'          : hyp_eg2['use_reg_bias_dict'],
                                    'invd_index'            : hyp_eg2['invd_index'],
                                    'angle_index'           : hyp_eg2['angle_index'],
                                    'dihyd_index'           : hyp_eg2['dihyd_index'],
                                    },
       	       	      'training'   :{
                                    'auto_scaling'          : {
                                                               'x_mean'               :hyp_eg2['scale_x_mean'],
                                                               'x_std'                :hyp_eg2['scale_x_std'],
                                                               'energy_mean'          :hyp_eg2['scale_y_mean'],
                                                               'energy_std'           :hyp_eg2['scale_y_std'],
                                                              },
                                    'normalization_mode'    : hyp_eg2['normalization_mode'],
                                    'loss_weights'          : hyp_eg2['loss_weights'],
                                    'learning_rate'         : hyp_eg2['learning_rate'],
                                    'initialize_weights'    : hyp_eg2['initialize_weights'],
                                    'val_disjoint'          : hyp_eg2['val_disjoint'],
                                    'val_split'             : hyp_eg2['val_split'],
                                    'epo'                   : hyp_eg2['epo'],
                                    'batch_size'            : hyp_eg2['batch_size'],
                                    'epostep'               : hyp_eg2['epostep'],
                                    'step_callback'         : {
                                                               'use'                  : hyp_eg2['use_step_callback'],
                                                               'epoch_step_reduction' : hyp_eg2['epoch_step_reduction'],
                                                               'learning_rate_step'   : hyp_eg2['learning_rate_step'],
                                                              },
                                    'linear_callback'       : {
                                                               'use'                  : hyp_eg2['use_linear_callback'],
                                                               'learning_rate_start'  : hyp_eg2['learning_rate_start'],
                                                               'learning_rate_stop'   : hyp_eg2['learning_rate_stop' ],
                                                               'epomin'               : hyp_eg2['epomin'],
                                                              },
       	       	       	       	    'early_callback'        : {
                                                               'use'                  : hyp_eg2['use_early_callback'],
                                                               'epomin'               : hyp_eg2['epomin'],
                                                               'patience'             : hyp_eg2['patience'],
                                                               'max_time'             : hyp_eg2['max_time'],
                                                               'delta_loss'           : hyp_eg2['delta_loss'],
                                                               'loss_monitor'         : hyp_eg2['loss_monitor'],
                                                               'factor_lr'            : hyp_eg2['factor_lr'],
                                                               'learning_rate_start'  : hyp_eg2['learning_rate_start'],
                                                               'learning_rate_stop'   : hyp_eg2['learning_rate_stop' ],
                                                              },
       	       	       	       	    'exp_callback'          : {
                                                               'use'                  : hyp_eg2['use_exp_callback'],
                                                               'epomin'               : hyp_eg2['epomin'],
                                                               'factor_lr'            : hyp_eg2['factor_lr'],
                                                              },
       	       	                    },
       	       	      'plots'      :{
                                    'unit_energy'           : hyp_eg2['unit'][0],
       	       	       	       	    'unit_gradient'         : hyp_eg2['unit'][1],
       	       	                    },
                      }

        hyp_dict_nac2={
                      'general'    :{
                                    'model_type'            : hyp_nac2['model_type'],
                                    },
                      'model'      :{
                                    'atoms'                 : data['natom'],
                                    'states'                : data['nstate'], 
                                    'nn_size'               : hyp_nac2['nn_size'],
                                    'depth'                 : hyp_nac2['depth'],
                                    'activ'                 : {
                                                               'class_name' : hyp_nac2['activ'],
                                                               'config'     : {
                                                                               'alpha' : hyp_nac2['activ_alpha'],
                                                                              },
                                                              },
                                    'use_dropout'           : hyp_nac2['use_dropout'],
                                    'dropout'               : hyp_nac2['dropout'],
                                    'use_reg_activ'         : hyp_nac2['use_reg_activ_dict'],
                                    'use_reg_weight'        : hyp_nac2['use_reg_weight_dict'], 
                                    'use_reg_bias'          : hyp_nac2['use_reg_bias_dict'],
                                    'invd_index'            : hyp_nac2['invd_index'],
                                    'angle_index'           : hyp_nac2['angle_index'],
                                    'dihyd_index'           : hyp_nac2['dihyd_index'],
                                    },
       	       	      'training'   :{
                                    'auto_scaling'          : {
                                                               'x_mean'               :hyp_nac2['scale_x_mean'],
                                                               'x_std'                :hyp_nac2['scale_x_std'],
                                                               'nac_mean'             :hyp_nac2['scale_y_mean'],
                                                               'nac_std'              :hyp_nac2['scale_y_std'],
                                                              },
                                    'normalization_mode'    : hyp_nac2['normalization_mode'],
                                    'learning_rate'         : hyp_nac2['learning_rate'],
                                    'phase_less_loss'  	    : hyp_nac2['phase_less_loss'],
                                    'initialize_weights'    : hyp_nac2['initialize_weights'],
                                    'val_disjoint'          : hyp_nac2['val_disjoint'],
                                    'val_split'             : hyp_nac2['val_split'],
                                    'epo'                   : hyp_nac2['epo'],
                                    'pre_epo'               : hyp_nac2['pre_epo'],
                                    'batch_size'            : hyp_nac2['batch_size'],
                                    'epostep'               : hyp_nac2['epostep'],
                                    'step_callback'         : {'use'                  : hyp_nac2['use_step_callback'],
                                                               'epoch_step_reduction' : hyp_nac2['epoch_step_reduction'],
                                                               'learning_rate_step'   : hyp_nac2['learning_rate_step'],
                                                              },
                                    'linear_callback'       : {
                                                               'use'                  : hyp_nac2['use_linear_callback'],
                                                               'learning_rate_start'  : hyp_nac2['learning_rate_start'],
                                                               'learning_rate_stop'   : hyp_nac2['learning_rate_stop' ],
                                                               'epomin'               : hyp_nac2['epomin'],
                                                              },
       	       	       	       	    'early_callback'        : {
                                                               'use'                  : hyp_nac2['use_early_callback'],
                                                               'epomin'               : hyp_nac2['epomin'],
                                                               'patience'             : hyp_nac2['patience'],
                                                               'max_time'             : hyp_nac2['max_time'],
                                                               'delta_loss'           : hyp_nac2['delta_loss'],
                                                               'loss_monitor'         : hyp_nac2['loss_monitor'],
                                                               'factor_lr'            : hyp_nac2['factor_lr'],
                                                               'learning_rate_start'  : hyp_nac2['learning_rate_start'],
                                                               'learning_rate_stop'   : hyp_nac2['learning_rate_stop' ],
                                                              },
       	       	       	       	    'exp_callback'          : {
                                                               'use'                  : hyp_nac2['use_exp_callback'],
                                                               'epomin'               : hyp_nac2['epomin'],
                                                               'factor_lr'            : hyp_nac2['factor_lr'],
                                                              },
       	       	                    },
       	       	      'plots'      :{
                                    'unit_nac'              : hyp_nac2['unit'],
       	       	                    },
                      }

        hyp_dict_eg['retraining']   = hyp_dict_eg['training']
        hyp_dict_eg2['retraining']  = hyp_dict_eg2['training']
        hyp_dict_nac['retraining']  = hyp_dict_nac['training']
        hyp_dict_nac2['retraining'] = hyp_dict_nac2['training']

        hyp_dict_eg['retraining']['initialize_weights']   = False
        hyp_dict_eg2['retraining']['initialize_weights']  = False
        hyp_dict_nac['retraining']['initialize_weights']  = False
        hyp_dict_nac2['retraining']['initialize_weights'] = False

        ## prepare training data
        self.natom      = data['natom']
        self.nstate     = data['nstate']
        self.version    = variables_all['version']
        self.ncpu       = variables_all['control']['ml_ncpu']
        self.pred_data  = variables['pred_data']
        self.train_mode = variables['train_mode']
        self.shuffle    = variables['shuffle']
        self.eg_unit    = variables['eg_unit']
        self.nac_unit   = variables['nac_unit']

        ## retraining has some bug at the moment, do not use
        if self.train_mode not in ['training','retraining','resample']:
            self.train_mode = 'training'

        if id == None or id == 1:
            self.name   = f"NN-{title}"
        else:
            self.name   = f"NN-{title}-{id}"
        self.silent     = variables['silent']
        self.x          = data['xyz'][:,:,1:].astype(float)

        ## convert unit of energy and force. au or si. data are in au.
        if self.eg_unit == 'si':
            self.H_to_eV        = 27.211396132
            self.H_Bohr_to_eV_A = 27.211396132/0.529177249
            self.keep_eV        = 1
            self.keep_eVA       = 1
        else:
            self.H_to_eV        = 1
            self.H_Bohr_to_eV_A = 1
       	    self.keep_eV       	= 27.211396132
       	    self.keep_eVA      	= 27.211396132/0.529177249

        if   self.nac_unit == 'si':
            self.Bohr_to_A  = 0.529177249/27.211396132 # convert to eV/A
            self.keep_A     = 1
        elif self.nac_unit == 'au':
            self.Bohr_to_A  = 1                             # convert to Eh/B
            self.keep_A     = 0.529177249/27.211396132
        elif self.nac_unit == 'eha':
            self.Bohr_to_A  = 0.529177249                   # convert to Eh/A
            self.keep_A     = 1/27.211396132

        ## combine y_dict
        self.y_dict = {}
        if nn_eg_type > 0:
            y_energy = data['energy']*self.H_to_eV
            y_grad   = data['gradient']*self.H_Bohr_to_eV_A
            self.y_dict['energy_gradient'] = [y_energy,y_grad]
        if nn_nac_type > 0:
            y_nac    = data['nac']/self.Bohr_to_A
            self.y_dict['nac'] = y_nac

        ## check permuation map
        self.x,self.y_dict = PermuteMap(self.x,self.y_dict,permute,hyp_eg['val_split'])

        ## combine hypers
        self.hyper = {}
        if   nn_eg_type == 1:  # same architecture with different weight
            self.hyper['energy_gradient']  = hyp_dict_eg
        else:
       	    self.hyper['energy_gradient']=[hyp_dict_eg,hyp_dict_eg2]
        if   nn_nac_type == 1: # same architecture with different weight
       	    self.hyper['nac']=hyp_dict_nac
       	elif nn_nac_type >  1:
            self.hyper['nac']=hyp_dict_nac=[hyp_dict_nac,hyp_dict_nac2]

        ## setup GPU list
        self.gpu_list={}
        if   gpu == 1:
            self.gpu_list['energy_gradient']=[0,0]
            self.gpu_list['nac']=[0,0]
       	elif gpu == 2:
            self.gpu_list['energy_gradient']=[0,1]
            self.gpu_list['nac']=[0,1]
       	elif gpu == 3:
            self.gpu_list['energy_gradient']=[0,1]
            self.gpu_list['nac']=[2,2]
       	elif gpu == 4:
            self.gpu_list['energy_gradient']=[0,1]
            self.gpu_list['nac']=[2,3]

        ## initialize model
        if   modeldir == None or id not in [None,1]:
            self.model = NeuralNetPes(self.name)
        else:
            self.model = NeuralNetPes(modeldir)

    def _heading(self):

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |                  Neural Networks                  |
 |                                                   |
 *---------------------------------------------------*

""" % (self.version)
 
       	return headline

    def _whatistime(self):
        return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    def _howlong(self,start,end):
        walltime=end-start
        walltime='%5d days %5d hours %5d minutes %5d seconds' % (int(walltime/86400),int((walltime%86400)/3600),int(((walltime%86400)%3600)/60),int(((walltime%86400)%3600)%60))
        return walltime

    def train(self):
        ## ferr      : dict
        ##            Fitting errors, share the same keys as y_dict

        start=time.time()

        self.model.create(self.hyper)

        topline='Neural Networks Start: %20s\n%s' % (self._whatistime(),self._heading())
        runinfo="""\n  &nn fitting \n"""

        if self.silent == 0:
            print(topline)
            print(runinfo)

        log=open('%s.log' % (self.name),'w')
        log.write(topline)
        log.write(runinfo)
        log.close()

        if self.train_mode == 'resample':
            out_index,out_errr,out_fiterr,out_testerr=self.model.resample(self.x,self.y_dict,gpu_dist=self.gpu_list,proc_async=self.ncpu>=4)
        else:
            ferr=self.model.fit(self.x,self.y_dict,gpu_dist=self.gpu_list,proc_async=self.ncpu>=4,fitmode=self.train_mode,random_shuffle=self.shuffle)
            print(ferr)
            #self.model.save()
            err_eg1=ferr['energy_gradient'][0]
            err_eg2=ferr['energy_gradient'][1]
            if 'nac' in ferr.keys():
                err_n=ferr['nac']
            else:
                err_n=np.zeros(2)

            train_info="""
  &nn validation mean absolute error
-------------------------------------------------------
      energy       gradient       nac(interstate)
        eV           eV/A         eV/A
  %12.8f %12.8f %12.8f
  %12.8f %12.8f %12.8f

""" % (err_eg1[0]*self.keep_eV, err_eg1[1]*self.keep_eVA, err_n[0]/self.keep_A ,err_eg2[0]*self.keep_eV, err_eg2[1]*self.keep_eVA, err_n[1]/self.keep_A)

        end=time.time()
        walltime=self._howlong(start,end)
        endline='Neural Networks End: %20s Total: %20s\n' % (self._whatistime(),walltime)

        if self.silent == 0:
            print(train_info)
            print(endline)

        log=open('%s.log' % (self.name),'a')
        log.write(train_info)
        log.write(endline)
        log.close()

        return self

    def load(self):
        self.model.load()

        return self

    def	appendix(self,addons):
       	## fake	function does nothing

       	return self

    def evaluate(self,x):
        ## y_pred   : dict
        ## y_std    : dict
        ##            Prediction and std, share the same keys as y_dict

        if x == None:
            with open('%s' % self.pred_data,'r') as preddata:
                pred=json.load(preddata)
            pred_natom,pred_nstate,pred_xyz,pred_invr,pred_energy,pred_gradient,pred_nac,pred_ci,pred_mo=pred
            x=np.array(pred_xyz)[:,:,1:].astype(float)
            y_pred,y_std=self.model.predict(x)
            entry=len(x)
        else:
            atoms=len(x)
            x=np.array(x)[:,1:4].reshape([1,atoms,3]).astype(float)
            y_pred,y_std=self.model.call(x)
            entry=1

        e_pred=y_pred['energy_gradient'][0]/self.H_to_eV
        g_pred=y_pred['energy_gradient'][1]/self.H_Bohr_to_eV_A
        e_std=y_std['energy_gradient'][0]/self.H_to_eV 
        g_std=y_std['energy_gradient'][1]/self.H_Bohr_to_eV_A
        if 'nac' in y_pred.keys():
            n_pred=y_pred['nac']*self.Bohr_to_A
            n_std=y_std['nac']*self.Bohr_to_A
        else:
            n_pred=np.zeros([entry,int(self.nstate*(self.nstate-1)/2),self.natom,3])
            n_std=np.zeros([entry,int(self.nstate*(self.nstate-1)/2),self.natom,3])

        if entry > 1 and self.silent == 0:
            de=np.abs(np.array(pred_energy)   - e_pred)
            dg=np.abs(np.array(pred_gradient) - g_pred)
            dn=np.abs(np.array(pred_nac)      - n_pred)
            for i in range(len(x)):
                print('%5s: %s %s %s' % (i+1,' '.join(['%8.4f' % (x) for x in de[i]]),' '.join(['%8.4f' % (np.amax(x)) for x in dg[i]]),' '.join(['%8.4f' % (np.amax(x)) for x in dn[i]])))

        ## Here I will need some function to print/save output
        length=len(x)
        if self.silent == 0:
            o=open('%s-e.pred.txt' % (self.name),'w')
            p=open('%s-g.pred.txt' % (self.name),'w')
            q=open('%s-n.pred.txt' % (self.name),'w')
            np.savetxt(o,np.concatenate((e_pred.reshape([length,-1]),e_std.reshape([length,-1])),axis=1))
            np.savetxt(p,np.concatenate((g_pred.reshape([length,-1]),g_std.reshape([length,-1])),axis=1))
            np.savetxt(q,np.concatenate((n_pred.reshape([length,-1]),n_std.reshape([length,-1])),axis=1))
            o.close()
            p.close()
            q.close()

        ## in MD, the prediction shape is (1,states) for energy and (1,states,atoms,3) for forces and nacs
        ## the return data should remove the batch size 1, thus take [0] of the data
        return {
                'energy'   : e_pred[0],
                'gradient' : g_pred[0],
                'nac'      : n_pred[0],
                'civec'    : None,
                'movec'    : None,
                'err_e'    : np.amax(e_std[0]),
                'err_g'    : np.amax(g_std[0]),
                'err_n'    : np.amax(n_std[0]),
       	       	}


