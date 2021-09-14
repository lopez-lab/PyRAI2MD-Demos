## Interface to Molcas for PyRAIMD
## Jingbai Li Feb 24 2020
## fix bug in reading nac data Jingbai Li Jun 22 2020

import os,subprocess,shutil,h5py
import numpy as np

from tools import Printcoord,NACpairs,whatistime,S2F,Markatom

class MOLCAS:
    ##This function run Molcas single point calculation
    ##The class works with OpenMolcas 19.11

    def __init__(self,variables_all,id=None):
        """
    Name               Type     Descriptions
    -----------------------------------------
    variables_all      dict     input file keywords from entrance.py.
    self.*:
        natom          int      number of atoms.
        ci             int      number of state, like ciroot in OpenMolcas.
        previous_civec list     list of configuration interaction vectors, used to do phase correction based on ci overlap. The shape depends from Molcas rasscf.h5 file.
        previous_movec list     list of molecular orbital vectors, used to do phase correction based on ci overlap. The shape depends from Molcas rasscf.h5 file.
        keep_tmp       int      keep the Molcas calculation folders (1) or not (0).
        verbose        int      print level.
        track_phase    int      track NAC phase using CAS(2,2). Future will add correction based on NAC overlap.
        basis          int      use customized basis set (1) or not(2).
        project        str      calculation name.
        workdir        str      Molcas workdir, molcas use this to write tmp files.
        calcdir        str      calculation folder.
        molcas         str      Molcas enviroment variable, executable folder.
        molcas_nproc   int      Molcas_nproc enviroment variable,number of threads for OMP parallelization.
        molcas_mem     int      Molcas_mem enviroment variable, calculation memory in MB.
        molcas_print   int      Molcas_print enviroment variable, print level.
        threads        int      number of threads for OMP parallelization.
        read_nac       int      read NAC (1) or not(2).
        use_hpc        int      use HPC (1) for calculation or not(0), like SLURM.
        """

        self.natom          = 0
        variables           = variables_all['molcas']
        self.ci             = variables['ci']
        self.previous_civec = variables['previous_civec']
        self.previous_movec = variables['previous_movec']
        self.keep_tmp       = variables['keep_tmp']
        self.verbose        = variables['verbose']
        self.track_phase    = variables['track_phase']
        self.basis          = variables['basis']

        self.project        = variables['molcas_project']
        self.workdir        = variables['molcas_workdir']
        self.calcdir        = variables['molcas_calcdir']
        self.molcas         = variables['molcas']
        self.molcas_nproc   = variables['molcas_nproc']
        self.molcas_mem     = variables['molcas_mem']
        self.molcas_print   = variables['molcas_print']
        self.threads        = variables['omp_num_threads']
        self.read_nac       = variables['read_nac']
        self.use_hpc        = variables['use_hpc']

        ## check calculation folder
        ## add index when running in adaptive sampling

        if id != None:
            self.calcdir    = '%s/tmp_MOLCAS-%s' % (self.calcdir,id)
        else:
            self.calcdir    = '%s/tmp_MOLCAS' % (self.calcdir)

        ## check Molcas workdir
        ## use local scratch /srv/tmp or /tmp if set to 'AUTO'

        if   self.workdir == 'AUTO':
            if os.path.exists('/srv/tmp') == True:
                self.workdir= '/srv/tmp/%s/%s/%s' % (os.environ['USER'],self.calcdir.split('/')[-2],self.calcdir.split('/')[-1])
            else:
                self.workdir= '/tmp/%s/%s/%s' % (os.environ['USER'],self.calcdir.split('/')[-2],self.calcdir.split('/')[-1])
        elif self.workdir == None:
            self.workdir    = self.calcdir

        ## set environment variables
        os.environ['MOLCAS_PROJECT']  = self.project   # the input name is fixed!
        os.environ['MOLCAS']          = self.molcas
        os.environ['MOLCAS_NPROC']    = self.molcas_nproc
        os.environ['MOLCAS_MEM']      = self.molcas_mem
        os.environ['MOLCAS_PRINT']    = self.molcas_print
        os.environ['MOLCAS_WORKDIR']  = self.workdir
        os.environ['OMP_NUM_THREADS'] = self.threads

        ## add gradient and nac part in the input, this eunsure correct order of computing them, users don't need to add them in the input

        alaska = ''
        i=int((self.ci-1)*self.ci/2)
        pairs=NACpairs(self.ci)
        for j in range(i):
            if self.read_nac == 1:
                alaska+='&ALASKA\nNAC=%s %s\n' % (pairs[j+1][0],pairs[j+1][1])
        for j in range(self.ci):
            alaska+='&ALASKA\nROOT=%s\n' % (j+1)
#        alaska+='&CASPT2\nSHIFT\n0.1\nXMULTistate\nall\nmaxiter\n1000'

        ## read input template from current directory
        ## make calculation folder and input file

        with open('%s.molcas' % (self.project),'r') as template:
            input=template.read()
        si_input ='%s\n%s' % (input,alaska)

        if os.path.exists(self.calcdir) == False:
            os.makedirs(self.calcdir)

        if os.path.exists(self.workdir) == False:
            os.makedirs(self.workdir)

        with open('%s/%s.inp' % (self.calcdir,self.project),'w') as out:
            out.write(si_input)

    def _setup_hpc(self):
        ## setup calculation using HPC
        ## read slurm template from .slurm files

        if os.path.exists('%s.slurm' % (self.project)) == True:
            with open('%s.slurm' % (self.project)) as template:
                submission=template.read()
        else:
            submission=''
        submission+="""
export INPUT=%s
export WORKDIR=%s
export MOLCAS_NPROCS=%s
export MOLCAS_MEM=%s
export MOLCAS_PRINT=%s
export OMP_NUM_THREADS=%s
export MOLCAS=%s
export PATH=$MOLCAS/bin:$PATH

if [ -d "/srv/tmp" ]
then
 export LOCAL_TMP=/srv/tmp
else
 export LOCAL_TMP=/tmp
fi

export MOLCAS_PROJECT=$INPUT
export MOLCAS_WORKDIR=$LOCAL_TMP/$USER/$SLURM_JOB_ID
mkdir -p $MOLCAS_WORKDIR/$MOLCAS_PROJECT

cd $WORKDIR
$MOLCAS/bin/pymolcas -f $INPUT.inp -b 1
rm -r $MOLCAS_WORKDIR/$MOLCAS_PROJECT
""" % (self.project,\
               self.calcdir,\
               self.molcas_nproc,\
               self.molcas_mem,\
               self.molcas_print,\
               self.threads,\
               self.molcas)

        with open('%s/%s.sbatch' % (self.calcdir,self.project),'w') as out:
            out.write(submission)


    def _setup_molcas(self,x):
        ## prepare .xyz .StrOrb files

        self.natom=len(x)

        ## prepare a list for marking atoms with different basis sets if necessary

        marks=[]
        if os.path.exists('%s.basis' % (self.project)) == True:
            with open('%s.basis' % (self.project)) as atommarks:
                marks=atommarks.read().splitlines()
                natom=int(marks[0])
                marks=marks[2:2+natom]

        if self.basis == 1 and len(marks)>0:            
            x=Markatom(x,marks,'molcas')

        ## save xyz and orbital files

        with open('%s/%s.xyz' % (self.calcdir,self.project),'w') as out:
            xyz='%s\n\n%s' % (self.natom,Printcoord(x))
            out.write(xyz)

        if   os.path.exists('%s.StrOrb' % (self.project)) == True and os.path.exists('%s/%s.RasOrb' % (self.calcdir,self.project)) == False:
            shutil.copy2('%s.StrOrb' % (self.project),'%s/%s.StrOrb' % (self.calcdir,self.project))
        elif os.path.exists('%s.StrOrb' % (self.project)) == True and os.path.exists('%s/%s.RasOrb' % (self.calcdir,self.project)) == True:
            shutil.copy2('%s/%s.RasOrb' % (self.calcdir,self.project),'%s/%s.StrOrb' % (self.calcdir,self.project))
        else:
            print('Molcas: missing guess orbital .StrOrb ')
            exit()

    def _run_molcas(self):
        ## run molcas calculation

        maindir=os.getcwd()
        os.chdir(self.calcdir)
        if self.use_hpc == 1:
            subprocess.run('sbatch -W %s/%s.sbatch' % (self.calcdir,self.project),shell=True)
        else:
            subprocess.run('%s/bin/pymolcas -f %s/%s.inp -b 1' % (self.molcas,self.calcdir,self.project),shell=True)
            subprocess.run('cp %s/%s/*.h5      %s' % (self.workdir,self.project,self.calcdir),shell=True)  # sometimes, Molcas doesn't copy these orbital files
       	    subprocess.run('cp %s/%s/*Orb*     %s' % (self.workdir,self.project,self.calcdir),shell=True)
       	    subprocess.run('cp %s/%s/*molden*  %s' % (self.workdir,self.project,self.calcdir),shell=True)
            shutil.rmtree('%s/%s' % (self.workdir,self.project))
        os.chdir(maindir)

    def _read_molcas(self):
        ## read molcas logfile and pack data

        with open('%s/%s.log' % (self.calcdir,self.project),'r') as out:
            log  = out.read().splitlines()
        natom    = self.natom
        casscf   = []
        gradient = []
        nac      = []
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

        ## pack data
        ## energy only includes the requested states by self.ci
        
        energy   = np.array(casscf)[0:self.ci]
        gradient = np.array(gradient)
        nac      = np.array(nac)

        ## clean movec and civec to reduce data size if phase correction is not requested
        if self.track_phase == 0:
            movec = np.zeros(0)
            civec = np.zeros(0)
        else:
            h5data   = h5py.File('%s/%s.rasscf.h5' % (self.calcdir,self.project),'r')
            civec    = np.array(h5data['CI_VECTORS'][()])
            movec    = np.array(h5data['MO_VECTORS'][()])
            norb     = int(len(movec)**0.5)
            movec    = movec[inactive*norb:(inactive+active)*norb]
            movec    = np.array(movec).reshape([active,norb])

        ## if NAC is not requested, always clean nac and reshape to [1,self.natom,3]

        if self.read_nac != 1:
            nac  = np.zeros([1,self.natom,3])

        return energy,gradient,nac,civec,movec

    def _phase_correction(self,x,nac,civec,movec):
        ## decide wheter to follow previous or search in data set
        ## current implementation only support cas(2,2)
        ## in other cases use phase_less_loss instead

        if   self.previous_civec is None and self.previous_movec is None:
            ## ci and mo of this geometry will be used as reference and do not apply phase correction 
            refmo=movec
            refci=civec
        elif self.previous_civec is not None and self.previous_movec is not None:
       	    ## apply phase correction based on the ci and mo of previous geometry
            refmo=self.previous_movec
            refci=self.previous_civec

        mooverlap,mophase,mofactor=self._mo_sign_correction(refmo,movec)
        cioverlap,ciphase,cifactor=self._ci_sign_correction(refci,civec,mofactor)
        nac   = (nac.T*cifactor).T
        civec = ((civec*mofactor).T*ciphase).T
        movec = (movec.T*mophase).T

        return nac,civec,movec

    def _mo_sign_correction(self,ref,mo):
        ## correct mo sign by overlap. assume the mo order is the same

        mooverlap = np.sum(mo*ref,axis=1)
        mophase   = np.sign(mooverlap)
        mofactor  = np.sign([1,mooverlap[0]*mooverlap[1],1])

        return mooverlap,mophase,mofactor

    def _ci_sign_correction(self,ref,ci,mofactor):
        ## correct mo sign by overlap.

        cioverlap = []
        nstate    = len(ref)
        ciphase   = np.ones(nstate)
        S         = np.array([np.sum(x*mofactor*ref,axis=1) for x in ci]) # sum of ci overlap matrix with mo factor
        aS        = np.abs(S)                             # absolute of sum of ci overlap matrix
        for i in range(nstate):                  # loop over nstate
            s=np.amax(aS[i])                     # find maximum overlap
            state=np.argmax(aS[i])               # find maximum overlap
            ciphase[i]=S[i][state]/aS[i][state]  # compute total sign
            aS[:,state]=0                        # clear the overlap of selected state
            cioverlap.append(S[i][state])

        cifactor  = []
        for i in range(nstate):
            for j in range(i+1,nstate):
                cifactor.append(ciphase[i]*ciphase[j])

        return cioverlap,ciphase,np.array(cifactor)

    def appendix(self,addons):
        ## appendix function to add ci and mo vectors of the previous step

        self.previous_civec = addons['pciv']
        self.previous_movec = addons['pmov']
        return self

    def evaluate(self,x):
        ## main function to run Molcas calculation and communicate with other PyRAIMD modules

        ## setup Molcas calculation
        self._setup_molcas(x)

        ## setup HPC settings
        if self.use_hpc == 1:
            self._setup_hpc()

        ## run Molcas calculation
        self._run_molcas()

        ## read Molcas output files
        energy,gradient,nac,civec,movec=self._read_molcas()

        ## phase correction
        if self.track_phase == 1:
            nac,civec,movec=self._phase_correction(x,nac,civec,movec)

        ## clean up
        if self.keep_tmp == 0:
            shutil.rmtree(self.calcdir)

        return {
                'energy'   : energy,
                'gradient' : gradient,
                'nac'	   : nac,
                'civec'    : civec,
                'movec'    : movec,
                'err_e'    : None,
                'err_g'    : None,
                'err_n'    : None,
                }

    def train(self):
        ## fake function

        return self

    def load(self):
        ## fake function

        return self

