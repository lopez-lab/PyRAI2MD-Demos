## The main function of PyRAIMD
## the first version  Jingbai Li Feb 16 2020
## the second version Jingbai Li May 14 2020
## the class version  Jingbai Li Jul 15 2020

import sys,os,time,datetime
import numpy as np
from tools import whatistime,howlong
from tools import Readcoord,Printcoord,Readinitcond
from entrance import ReadInput,StartInfo
from methods import QM
from aimd import AIMD
from hybrid import MIXAIMD
from dynamixsampling import Sampling
from adaptive_sampling import AdaptiveSampling

def logo(version):

    credits="""
  --------------------------------------------------------------
                               *
                              /+\\
                             /+++\\
                            /+++++\\
                           /PyRAIMD\\
                          /+++++++++\\
                         *===========*

                          Python Rapid
                     Artificial Intelligence
                  Ab Initio Molecular Dynamics

                      Developer @Jingbai Li
                  Northeastern University, USA

                          version:   %s


  With contriutions from (in alphabetic order):
    Andre Eberhard	       - Gaussian process regression
    Jingbai Li/Daniel Susman   - Zhu-Nakamura surface hopping
    Jingbai Li                 - Fewest switches surface hopping
                                 Velocity Verlet
                                 Interface to OpenMolcas/BAGEL
                                 Adaptive sampling (with enforcement)
                                 QC/ML non-adiabatic molecular dynamics
    Patrick Reiser	       - Neural networks (pyNNsMD)

  Special acknowledgement to:
    Steven A. Lopez	       - Project directorship
    Pascal Friederich          - ML directoriship

""" % (version)

    return credits

class PYRAIMD:

    def __init__(self,input):
        version='0.9b'
        if input == None: print(logo(version)), exit()
        input_dict=open(input,'r').read().split('&')
        self.variables_all=ReadInput(input_dict)
        input_info=StartInfo(self.variables_all)
        self.variables_all['version']                  = self._version_info(version,input_info)

    def _version_info(self,x,y):

        ## x: float
        ##    Version 
        ## y: str
        ##    Input information

        info="""%s

%s

""" % (logo(x),y)
        return info

    def _machine_learning(self):
        jobtype  = self.variables_all['control']['jobtype']
        qm       = self.variables_all['control']['qm'] 

        model=QM(qm,self.variables_all,id=None)

        if   jobtype == 'train':
            model.train()
        elif jobtype == 'prediction':
            model.load()
            model.evaluate(None)  # None will use json file

    def _dynamics(self):
        title    = self.variables_all['control']['title']
        qm       = self.variables_all['control']['qm']
        md       = self.variables_all['md']
        initcond = md['initcond']
        nesmb    = md['nesmb']
        method   = md['method']
        format   = md['format']
        gl_seed  = md['gl_seed']
        temp     = md['temp']
        if initcond == 0:
            ## load initial condition from .xyz and .velo
            xyz,M=Readcoord(title)
            velo=np.loadtxt('%s.velo' % (title))
        else:
            ## use sampling method to generate intial condition
            trvm=Sampling(title,nesmb,gl_seed,temp,method,format)[-1]
            xyz,mass,velo=Readinitcond(trvm)
            ## save sampled geometry and velocity
            initxyz_info='%d\n%s\n%s' % (len(xyz),'%s sampled geom %s at %s K' % (method,nesmb,temp),Printcoord(xyz))
            initxyz=open('%s.xyz' % (title),'w')
            initxyz.write(initxyz_info)
            initxyz.close()
            initvelo=open('%s.velo' % (title),'w')
            np.savetxt(initvelo,velo,fmt='%30s%30s%30s')
            initvelo.close()
        method=QM(qm,self.variables_all,id=None)
        method.load()
        traj=AIMD(self.variables_all,QM=method,id=None,dir=None)
        traj.run(xyz,velo)

    def _hybrid_dynamics(self):
        title    = self.variables_all['control']['title']
        qm	 = self.variables_all['control']['qm']
        abinit   = self.variables_all['control']['abinit']
        md	 = self.variables_all['md']
        initcond = md['initcond']
        nesmb    = md['nesmb']
        method   = md['method']
        format   = md['format']
        gl_seed  = md['gl_seed']
        temp     = md['temp']
        if initcond == 0:
            ## load initial condition from .xyz and .velo
            xyz,M=Readcoord(title)
            velo=np.loadtxt('%s.velo' % (title))
        else:
            ## use sampling method to generate intial condition
            trvm=Sampling(title,nesmb,gl_seed,temp,method,format)[-1]
            xyz,mass,velo=Readinitcond(trvm)
            ## save sampled geometry and velocity
            initxyz_info='%d\n%s\n%s' % (len(xyz),'%s sampled geom %s at %s K' % (method,nesmb,temp),Printcoord(xyz))
            initxyz=open('%s.xyz' % (title),'w')
            initxyz.write(initxyz_info)
            initxyz.close()
            initvelo=open('%s.velo' % (title),'w')
            np.savetxt(initvelo,velo,fmt='%30s%30s%30s')
            initvelo.close()
        ref=QM(abinit,self.variables_all,id=None)
        ref.load()
        method=QM(qm,self.variables_all,id=None)
        method.load()
        traj=MIXAIMD(self.variables_all,QM=method,REF=ref,id=None,dir=None)
        traj.run(xyz,velo)

    def	_active_search(self):
        sampling=AdaptiveSampling(self.variables_all)
        sampling.search()

    def run(self):
        jobtype = self.variables_all['control']['jobtype']
        job_func={
        'md'         : self._dynamics,
        'hybrid'     : self._hybrid_dynamics,
        'adaptive'   : self._active_search,
        'train'      : self._machine_learning,
        'prediction' : self._machine_learning,
        ## 'search'  : self._machine_learning,  ## not implemented
        }

        job_func[jobtype]()


if __name__ == '__main__':

    if 'PYRAIMD' in os.environ.keys(): os.environ['PYTHONPATH'] = os.environ['PYRAIMD']
    pmd=PYRAIMD

    if len(sys.argv) < 2:
        print('\n  PyRAIMD: no input file...')
        pmd(None)
    else:
        pmd(sys.argv[1]).run()
