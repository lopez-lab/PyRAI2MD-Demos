## active search for PyRAIMD
## Jingbai Li Jul 11 2020

import time,datetime,json,shutil
import multiprocessing,os
from multiprocessing import Pool
import numpy as np
from aimd import AIMD
from methods import QM
from data_processing import AddTrainData,GetInvR
from tools import Printcoord,Readinitcond
from aligngeom import AlignGeom
from dynamixsampling import Sampling
from periodic_table import BondLib

class AdaptiveSampling:

    def __init__(self,variables_all):

        ## variables for generating initcond
        md                = variables_all['md']
        initcond          = md['initcond']
        nesmb             = md['nesmb']
        method            = md['method']
        format            = md['format']
        gl_seed           = md['gl_seed']
        temp              = md['temp']

        ## variables for controling workflow
        control           = variables_all['control']
        self.version      = variables_all['version']
        self.qm           = control['qm']
        self.abinit       = control['abinit']
        self.ml_ncpu      = control['ml_ncpu']
        self.qc_ncpu      = control['qc_ncpu']
        self.title        = control['title']
        self.maxiter      = control['maxiter']
        self.refine       = control['refine']
       	self.refine_num   = control['refine_num']
       	self.refine_start = control['refine_start']
       	self.refine_end   = control['refine_end']
       	self.load         = control['load']
        self.transfer     = control['transfer']
        self.pop_step     = control['pop_step']
        self.variables = variables_all.copy() # hard copy all input variables, so I can change them safely
        self.threshold = {
        'maxsample'    : control['maxsample'],
        'maxenergy'    : control['maxenergy'],
        'minenergy'    : control['minenergy'],
        'maxgradient'  : control['maxgradient'],
        'mingradient'  : control['mingradient'],
        'maxnac'       : control['maxnac'],
        'minnac'       : control['minnac'],
        'neighbor'     : control['neighbor'],
        }

       	## variables for checking QC results
       	self.ci	       	  = md['ci']
       	self.read_nac     = variables_all[self.abinit]['read_nac']

        ## trajectories properties
        self.iter         = 0                            # number of search iteration
        self.ntraj        = nesmb                        # number of trajectories
        self.initcond     = [[] for x in range(nesmb)]   # initial conditions
        self.selec_geo    = [[] for x in range(nesmb)]   # selected geometries for QC calculation
        self.discard_geo  = [[] for x in range(nesmb)]   # discarded geometries before QC calculation
        self.selec_e      = [[] for x in range(nesmb)]   # error of energy
        self.index_e      = [[] for x in range(nesmb)]   # index of selected geometry based on energy error
        self.selec_g      = [[] for x in range(nesmb)]   # error of gradient
        self.index_g      = [[] for x in range(nesmb)]   # index of selected geometry based on gradient error
        self.selec_n      = [[] for x in range(nesmb)]   # error of nac
        self.index_n      = [[] for x in range(nesmb)]   # index of selected geometry based on nac error

        np.random.seed(gl_seed)
        trvm=Sampling(self.title,nesmb,gl_seed,temp,method,format)
        for ntraj,x in enumerate(trvm):
            xyz,M,V=Readinitcond(x)
            self.initcond[ntraj]=[xyz,V]

    def _run_aimd(self):
        ## wrap variables for multiprocessing
        variables_wrapper=[[n,x[0],x[1]]for n,x in enumerate(self.initcond)]
        ntraj=len(variables_wrapper)

        ## adjust multiprocessing if necessary
        ncpu = np.amin([ntraj,self.ml_ncpu])

        md_traj=[[] for x in range(ntraj)]
        ## start multiprocessing
        pool=multiprocessing.Pool(processes=ncpu)
        for val in pool.imap_unordered(self._aimd_wrapper,variables_wrapper):
            traj_id,md_hist=val
            md_traj[traj_id]=md_hist
        pool.close()
        return md_traj

    def _aimd_wrapper(self,initial_condition):
        ## run AIMD
        ## multiprocessing doesn't support shared-memory
        ## load mode in each worker process here :(
        qm=QM(self.qm,self.variables,id=self.iter)
        qm.load()
        traj_id,xyz,velo=initial_condition
        traj=AIMD(self.variables,QM=qm,id=traj_id+1,dir=True)
        md_hist=traj.run(xyz,velo)
        return traj_id,md_hist

    def _run_abinit(self):
        ## wrap variables for multiprocessing
        geom=[]
        for xyz in self.selec_geo:  # flatten the geometry list
            geom+=xyz
        variables_wrapper=[[n,xyz]for n,xyz in enumerate(geom)]
        ngeom=len(variables_wrapper)
        ## adjust multiprocessing if necessary
        ncpu = np.amin([ngeom,self.qc_ncpu])

        ## start multiprocessing
        qc_results=[[] for x in range(ngeom)]
        pool=multiprocessing.Pool(processes=ncpu)
        for val in pool.imap_unordered(self._abinit_wrapper,variables_wrapper):
            geom_id,xyz,energy,gradient,nac,civec,movec=val
            qc_results[geom_id]=[xyz,energy.tolist(),gradient.tolist(),nac.tolist(),civec.tolist(),movec.tolist()]
        pool.close()

        ## check qc results and exclude non-converged ones
        results=[]
        for i in qc_results:
            if   self.read_nac == 1 and len(i[1]) == self.ci and len(i[2]) == self.ci and len(i[3]) == self.ci*(self.ci-1)/2:
                results.append(i)
            elif self.read_nac == 0 and len(i[1]) == self.ci and len(i[2]) == self.ci:
                results.append(i)

        return results

    def _abinit_wrapper(self,selec_geom):
        ## run QC calculation
        geom_id,xyz=selec_geom
        ## the geometry alignment is not necessary if NAC is not request. Maybe add a condition statement in the future
        geom_pool=self.variables[self.qm]['data'][2]
        choose=np.random.choice(np.arange(len(geom_pool)),np.amin([50,len(geom_pool)]),replace=False)
        geom_pool=np.array(geom_pool)[choose]
        similar,rmsd_min=AlignGeom(xyz,geom_pool)
        movec=self.variables[self.qm]['data'][-1][similar]
        civec=self.variables[self.qm]['data'][-2][similar]
        addons={
        'pciv' : civec,
        'pmov' : movec,
        }

        qc=QM(self.abinit,self.variables,id=geom_id+1)
        qc.appendix(addons)
        results  = qc.evaluate(xyz)
        energy   = results['energy']
        gradient = results['gradient']
        nac      = results['nac']
        civec    = results['civec']
        movec    = results['movec']

        return geom_id,xyz,energy,gradient,nac,civec,movec

    def _screen_error(self,md_traj):
        ## check errors
        maxsample     = self.threshold['maxsample']
        neighbor      = self.threshold['neighbor']
        minerr_e      = self.threshold['minenergy']
        minerr_g      = self.threshold['mingradient']
        minerr_n      = self.threshold['minnac']

        checkpoint={
        'last'         : [],
        'geom'         : [],
        'energy'       : [],
        'gradient'     : [],
        'nac'          : [],
        'err_e'        : [],
        'err_g'        : [],
        'err_n'        : [],
        'minerr_e'     : minerr_e,
        'minerr_g'     : minerr_g,
        'minerr_n'     : minerr_n,
        'max_e'        : 0,
       	'max_g'	       : 0,
       	'max_n'	       : 0,
        'new_geom'     : [],
        'discard_geom' : [],
        'uncertain'    : [],
        'pop'          : [],
        }

        for ntraj in range(self.ntraj):
            last,geo,e,g,n,err_e,err_g,err_n,pop=np.array(md_traj[ntraj]).T

            ## pack data into checkpoing dict
            checkpoint['last'].append(last)            # the last MD step
            checkpoint['geom'].append(geo.tolist())    # all recorded geometries
            checkpoint['energy'].append(e.tolist())    # all energies
            checkpoint['gradient'].append(g.tolist())  # all forces
            checkpoint['nac'].append(n.tolist())       # all NACs
            checkpoint['err_e'].append(err_e.tolist()) # all prediction	error in energies
       	    checkpoint['err_g'].append(err_g.tolist()) # all prediction	error in forces
       	    checkpoint['err_n'].append(err_n.tolist()) # all prediction error in NACs
            checkpoint['pop'].append(pop.tolist())     # all populations

            if np.amax(err_e) > checkpoint['max_e']:
                checkpoint['max_e'] = np.amax(err_e)   # max prediction error in energies
       	    if np.amax(err_g) >	checkpoint['max_g']:
       	       	checkpoint['max_g'] = np.amax(err_g)   # max prediction	error in forces
       	    if np.amax(err_n) >	checkpoint['max_n']:
       	       	checkpoint['max_n'] = np.amax(err_n)   # max prediction error in NACs

            ## largest n std in e, g, and n
            #selec_e,index_e = self._localmax(err_e,maxsample,neighbor)
            #selec_g,index_g = self._localmax(err_g,maxsample,neighbor)
            #selec_n,index_n = self._localmax(err_n,maxsample,neighbor)

            ## find index of geometries exceeding the threshold of prediction error
            index_e = np.argwhere(err_e > minerr_e)
            selec_e = err_e[index_e.reshape(-1)]
            index_g = np.argwhere(err_g > minerr_g)           
       	    selec_g = err_g[index_g.reshape(-1)]
            index_n = np.argwhere(err_n > minerr_n)           
       	    selec_n = err_n[index_n.reshape(-1)]

            ##  merge index and remove duplicate in selec_geom
            index_tot=np.concatenate((index_e,index_g)).astype(int)
            index_tot=np.concatenate((index_tot,index_n)).astype(int)
            index_tot=np.unique(index_tot)

            ## record number of uncertain geometry before merging with refinement geometry
            checkpoint['uncertain'].append(len(index_tot))

            ## refine crossing region, optionally
            if self.refine == 1:
                e=np.array([np.array(x) for x in e])
                state=len(e[0])
                pair=int(state*(state-1)/2)
                gap_e=np.zeros([len(e),pair])  # initialize gap matrix
                pos=-1
                for i in range(state):         # compute gap per pair of states
                    for j in range(i+1,state):
                        pos+=1
                        gap_e[:,pos]=np.abs(e[:,i]-e[:,j])
                gap_e=np.amin(gap_e,axis=1)    # pick the smallest gap per point
                index_r = np.argsort(gap_e[self.refine_start:self.refine_end])[0:self.refine_num]
                index_tot=np.concatenate((index_tot,index_r)).astype(int)
                index_tot=np.unique(index_tot)

            keep_geo,discard_geo    = self._distance_filter(np.array(geo)[index_tot].tolist()) # filter out the unphyiscal geometries based on atom distances
            self.selec_geo[ntraj]   = keep_geo
            self.discard_geo[ntraj] = discard_geo
            self.selec_e[ntraj]     = selec_e
            self.index_e[ntraj]     = index_e
       	    self.selec_g[ntraj]     = selec_g
       	    self.index_g[ntraj]     = index_g
       	    self.selec_n[ntraj]     = selec_n
            self.index_n[ntraj]     = index_n

        checkpoint['new_geom']      = self.selec_geo        # new geometries
        checkpoint['discard_geom']  = self.discard_geo      # discarded geometries

        return checkpoint

    def _distance_filter(self,geom):
        ## This function filter out unphysical geometries based on atom distances
        keep=[]
        discard=[]
        if len(geom) > 0:
            natom=len(geom[0])
            for geo in geom:
                unphysical=0
                for i in range(natom):
                    for j in range(i+1,natom): 
                        atom1=geo[i][0]
                        coord1=np.array(geo[i][1:4])
       	       	       	atom2=geo[j][0]
                        coord2=np.array(geo[j][1:4])
                        distance=np.sum((coord1-coord2)**2)**0.5
                        threshld=BondLib(atom1,atom2)
                        if distance < threshld*0.7:
                            unphysical=1
                if unphysical == 1:
                    discard.append(geo) 
                else:
                    keep.append(geo)

        return keep,discard

    def _localmax(self,error,maxsample,neighbor):
        ## This function find local maximum of error as function of simulation step

        ## find all local maximum: g_1 gradient from n-1 to n; g_2 gradient from n to n+1
        index_lcmax=[]   # index for local maximum
        error_lcmax=[]   # local maximum errors
        for n,i in enumerate(error):
            if i == error[0]:
                g_1=1
            else:
                g_1=i-error[n-1]

            if i == error[-1]:
                g_2=0
            else:
                g_2=error[n+1]-i

            if g_1 >0 and g_2 <= 0:
                index_lcmax.append(n)
                error_lcmax.append(i)

        index_lcmax=np.array(index_lcmax)
        error_lcmax=np.array(error_lcmax)
        check_lcmax=np.ones(len(error_lcmax))

        ## only include non-neighbor
        index_error=[]
        selec_error=[]
        for n,i in enumerate(np.argsort(-error_lcmax)):
            if check_lcmax[i] == 1:
                index_error.append(index_lcmax[i])
                selec_error.append(error_lcmax[i])
                for j in np.argsort(-error_lcmax)[n+1:]:
                    if np.abs(index_lcmax[i]-index_lcmax[j]) < neighbor:
                        check_lcmax[j]=0

        index_error=np.array(index_error)
        selec_error=np.array(selec_error)
        ## adjust maxsample
        if len(selec_error) > maxsample:
            selec_error=selec_error[:maxsample]
            index_error=index_error[:maxsample]

        return selec_error,index_error

    def _update_train_set(self,results):
        data,postdata,data_info=AddTrainData(self.variables[self.qm],results,self.iter+1)
        self.variables[self.qm]['data']      = data
        self.variables[self.qm]['postdata']  = postdata
       	self.variables[self.qm]['data_info'] = data_info

    def _train_model(self):

        if self.iter == 1:
            ## set to do a fresh training in case you want to first train a model (self.load == 0)
            self.variables[self.qm]['train_mode'] = 'training'
        else:
            if self.transfer == 0:
            ## set to do a fresh training for the next iteraction
                self.variables[self.qm]['train_mode'] = 'training'
            else:
       	    ## set to do a transfer learning for the next iteraction
            ## copy previous model NN-(self.title)-(self.iter-1) to NN-(self.title)-(self.iter) as initial guess
                self.variables[self.qm]['train_mode'] = 'retraining'
                if self.iter == 2:
                    shutil.copytree('NN-%s' % (self.title),'NN-%s-%s' % (self.title,self.iter))
                else:
                    shutil.copytree('NN-%s-%s' % (self.title,self.iter-1),'NN-%s-%s' % (self.title,self.iter))

        if self.iter > 1 or self.load == 0:
            pool=multiprocessing.Pool(processes=1)
            for val in pool.imap_unordered(self._train_wrapper,[None]):
                val=None
            pool.close()

    def _train_wrapper(self,fake):
        model=QM(self.qm,self.variables,id=self.iter)
        model.train()
        return None

    def _checkpoint(self,checkpoint_dict):
        logpath      = os.getcwd()
        last         = checkpoint_dict['last']
        geom         = checkpoint_dict['geom']
        uncertain    = checkpoint_dict['uncertain']
        new_geom     = checkpoint_dict['new_geom']
        discard_geom = checkpoint_dict['discard_geom']
        err_e        = checkpoint_dict['err_e']
       	err_g  	     = checkpoint_dict['err_g']
       	err_n  	     = checkpoint_dict['err_n']
        max_e        = checkpoint_dict['max_e']
        max_g        = checkpoint_dict['max_g']
        max_n        = checkpoint_dict['max_n']
        minerr_e     = checkpoint_dict['minerr_e']
        minerr_g     = checkpoint_dict['minerr_g']
        minerr_n     = checkpoint_dict['minerr_n']
        pop          = checkpoint_dict['pop']

        converged    = 0
        refinement   = 0
        found        = 0
       	discarded    = 0
        all_geom     = 0
        traj_info='  &adaptive sampling progress\n'
        for i in range(self.ntraj):
            marker=''
            if uncertain[i] == 0:
                converged+=1
                marker='*'
            traj_info+='  Traj %6s: %8s steps found %8s new geometries discard %8s geometries => MaxErr(Energy: %8.4f Gradient: %8.4f NAC: %8.4f) %s\n' % (i+1,last[i][-1],uncertain[i],len(discard_geom[i]),np.amax(err_e[i]),np.amax(err_g[i]),np.amax(err_n[i]),marker)
            found+=uncertain[i]
            discarded+=len(discard_geom[i])
            all_geom+=len(new_geom[i])

        refinement=all_geom+discarded-found

        log_info="""
%s

  &adaptive sampling iter %s : converged %s of %s found %s new geometries added %s refinement discarded %s

                 the largest error
-------------------------------------------------------
  Energy:                   %8.4f   %8.4f  %6s
  Gradient:                 %8.4f   %8.4f  %6s
  Non-adiabatic coupling:   %8.4f   %8.4f  %6s

""" % (traj_info,self.iter, converged, self.ntraj,found,refinement,discarded,\
       max_e, minerr_e, max_e<=minerr_e,\
       max_g, minerr_g, max_g<=minerr_g,\
       max_n, minerr_n, max_n<=minerr_n)

        print(log_info)
        mdlog=open('%s/%s.log' % (logpath,self.title),'a')
        mdlog.write(log_info)
        mdlog.close()

        average_pop=[]
        for p in pop:
            if len(p) >= self.pop_step:
                average_pop.append(p[0:self.pop_step])
        if len(average_pop) > 0:
            average_pop=np.mean(average_pop,axis=0)
            pop_info=''
            for n,p in enumerate(average_pop):
                pop_info+='%-5s%s\n'% (n,' '.join(['%24.16f ' % (x) for x in p]))
            mdpop=open('%s/%s-%s.pop' % (logpath,self.title,self.iter),'w')
            mdpop.write(pop_info)
            mdpop.close()

        #savethis={self.iter:checkpoint_dict} ## This saves too much !!
        savethis={
        self.iter: {
        'new_geom' : checkpoint_dict['new_geom']
        }
        }
        if self.iter == 1:
            with open('%s/%s.adaptive.json' % (logpath,self.title), 'w') as outfile:
                json.dump(savethis,outfile)
        else:
            with open('%s/%s.adaptive.json' % (logpath,self.title), 'r') as infile:
                loadthis=json.load(infile)
            savethis.update(loadthis)
            with open('%s/%s.adaptive.json' % (logpath,self.title), 'w') as outfile:
                json.dump(savethis,outfile)

        return converged,refinement

    def _heading(self):

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |          Adaptive Sampling for ML-NAMD            |
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

    def search(self):
        logpath=os.getcwd()
        start=time.time()
        heading='Adaptive Sampling Start: %20s\n%s' % (self._whatistime(),self._heading())
        print(heading)
        mdlog=open('%s/%s.log' % (logpath,self.title),'w')
        mdlog.write(heading)
        mdlog.close()


        for iter in range(self.maxiter):
            self.iter=iter+1
            self._train_model()
            md_traj=self._run_aimd()
            checkpoint_dict=self._screen_error(md_traj)
            converged,refinement=self._checkpoint(checkpoint_dict)

            if self.iter > self.maxiter:
                break
            else:
                if converged == self.ntraj and refinement == 0:
                    break
                else:
                    results=self._run_abinit()
                    self._update_train_set(results)


        end=time.time()
        walltime=self._howlong(start,end)
        tailing='Adaptive Sampling End: %20s Total: %20s\n' % (self._whatistime(),walltime)
        print(tailing)
        mdlog=open('%s/%s.log' % (logpath,self.title),'a')
        mdlog.write(tailing)
        mdlog.close()

