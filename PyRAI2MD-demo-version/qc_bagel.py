## Interface to BAGEL for PyRAIMD
## Jingbai Li Sep 29 2020

import os,subprocess,shutil,h5py
import numpy as np

from tools import Printcoord,NACpairs,whatistime,S2F,NACpairs

class BAGEL:
    ## This function run BAGEL single point calculation

    def __init__(self,variables_all,id=None):
        """
    Name               Type     Descriptions
    -----------------------------------------
    variables_all      dict     input file keywords from entrance.py.
    self.*:
        natom          int      number of atoms.
        ci             int      number of state, like ciroot in OpenMolcas.
        previous_civec list     list of configuration interaction vectors, not used with BAGEL
        previous_movec list     list of molecular orbital vectors, not used with BAGEL
        keep_tmp       int      keep the BAGEL calculation folders (1) or not (0).
        verbose        int      print level.
        project        str      calculation name.
        workdir        str      calculation folder.
        bagel          str      BAGEL executable folder
        nproc          int      number of CPUs for parallelization
        mpi            str      path to mpi library
        blas           str      path to blas library
        lapack         str      path to lapack library
        boost          str      path to boost library
        mkl            str      path to mkl library
        arch           str      CPU architecture
        threads        int      number of threads for OMP parallelization.
        read_nac       int      read NAC (1) or not(2).
        use_hpc        int      use HPC (1) for calculation or not(0), like SLURM.
        use_mpi        int      use MPI (1) for calculation or not(0).
        """

        self.natom          = 0
        variables           = variables_all['bagel']
        self.keep_tmp       = variables['keep_tmp']
        self.verbose        = variables['verbose']
        self.ci             = variables['ci']
        self.project        = variables['bagel_project']
        self.workdir        = variables['bagel_workdir']
        self.archive        = variables['bagel_archive']
        self.bagel          = variables['bagel']
        self.nproc          = variables['bagel_nproc']
        self.mpi            = variables['mpi']
        self.blas      	    = variables['blas']
        self.lapack         = variables['lapack']
        self.boost          = variables['boost']
        self.mkl            = variables['mkl']
       	self.arch           = variables['arch']
        self.read_nac       = variables['read_nac']
        self.threads        = variables['omp_num_threads']
        self.use_mpi        = variables['use_mpi']
        self.use_hpc        = variables['use_hpc']

        ## check calculation folder
        ## add index when running in adaptive sampling

        if id != None:
            self.workdir    = '%s/tmp_BAGEL-%s' % (self.workdir,id)
        else:
            self.workdir    = '%s/tmp_BAGEL' % (self.workdir)

        ## set environment variables
        os.environ['BAGEL_PROJECT']       = self.project   # the input name is fixed!
        os.environ['BAGEL']               = self.bagel
        os.environ['BLAS']                = self.blas
        os.environ['LAPACK']              = self.lapack
        os.environ['BOOST']               = self.boost
        os.environ['BAGEL_WORKDIR']       = self.workdir
        os.environ['OMP_NUM_THREADS']     = self.threads
        os.environ['MKL_NUM_THREADS']     = self.threads
        os.environ['BAGEL_NUM_THREADS']   = self.threads
        os.environ['MV2_ENABLE_AFFINITY'] = '0'
        os.environ['LD_LIBRARY_PATH']     = '%s/lib:%s/lib:%s:%s:%s/lib:%s' % (self.mpi,self.bagel,self.blas,self.lapack,self.boost,os.environ['LD_LIBRARY_PATH'])
        os.environ['PATH']                = '%s/bin:%s' % (self.mpi,os.environ['PATH'])

    def _xyz2json(self,natom,coord):
        ## convert xyz from array to bagel format (Bohr)

        a2b=1/0.529177249   # angstrom to bohr
        jxyz=''
        comma=','

        for n,line in enumerate(coord):
            e,x,y,z=line
            if n == natom-1:
                comma=''
            jxyz+="""{ "atom" : "%s", "xyz" : [%f, %f, %f]}%s\n""" % (e,float(x)*a2b,float(y)*a2b,float(z)*a2b,comma)

        return jxyz

    def _setup_hpc(self):
        ## setup calculation using HPC
        ## read slurm template from .slurm files

        if os.path.exists('%s.slurm' % (self.project)) == True:
            with open('%s.slurm' % (self.project)) as template:
                submission=template.read()
        else:
            submission=''

        submission+="""
export BAGEL_PROJECT=%s
export BAGEL=%s
export BLAS=%s
export LAPACK=%s
export BOOST=%s
export MPI=%s
export BAGEL_WORKDIR=%s
export OMP_NUM_THREADS=%s
export MKL_NUM_THREADS=%s
export BAGEL_NUM_THREADS=%s
export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL/lib:$BALS:$LAPACK:$BOOST/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

source %s %s

cd $BAGEL_WORKDIR
""" % (self.project,\
               self.bagel,\
               self.blas,\
               self.lapack,\
               self.boost,\
               self.mpi,\
               self.workdir,\
               self.threads,\
               self.threads,\
               self.threads,\
               self.mkl,\
               self.arch)

        if self.use_mpi == 0:
            submission+='%s/bin/BAGEL %s/%s.json > %s/%s.log\n' % (self.bagel,self.workdir,self.project,self.workdir,self.project)
        else:
            submission+='mpirun -np %s %s/bin/BAGEL %s/%s.json > %s/%s.log\n' % (self.nproc,self.bagel,self.workdir,self.project,self.workdir,self.project)

        with open('%s/%s.sbatch' % (self.workdir,self.project),'w') as out:
            out.write(submission)

    def _setup_bagel(self,x):
        ## prepare .json .archive files

        self.natom=len(x)

        ## Read input template from current directory

        with open('%s.bagel' % (self.project),'r') as template:
            input=template.read().splitlines()

        part1=''
        part2=''
        breaker=0
        for line in input:
            if '******' in line:
                breaker = 1
                continue
            if breaker == 0:
                part1+='%s\n' % line
            else:
                part2+='%s\n' % line

        coord=self._xyz2json(self.natom,x)

        si_input = part1+coord+part2

        ## check BAGEL workdir
        if os.path.exists(self.workdir) == False:
            os.makedirs(self.workdir)

        ## save xyz file
        with open('%s/%s.json' % (self.workdir,self.project),'w') as out:
            out.write(si_input)

        ## save .archive file
        if   os.path.exists('%s.archive' % (self.project)) == False:
            print('BAGEL: missing guess orbital .archive ')
            exit()

        if self.archive == 'default':
            self.archive = self.project
        
        if os.path.exists('%s/%s.archive' % (self.workdir,self.archive)) == False:
            shutil.copy2('%s.archive' % (self.project),'%s/%s.archive' % (self.workdir,self.archive))

        ## clean calculation folder
        os.system("rm %s/ENERGY*.out > /dev/null 2>&1" % (self.workdir))
        os.system("rm %s/FORCE_*.out > /dev/null 2>&1" % (self.workdir))
        os.system("rm %s/NACME_*.out > /dev/null 2>&1" % (self.workdir))

    def _run_bagel(self):
        ## run BAGEL calculation

        maindir=os.getcwd()
        os.chdir(self.workdir)
        if self.use_hpc == 1:
            subprocess.run('sbatch -W %s/%s.sbatch' % (self.workdir,self.project),shell=True)
        else:
            if self.use_mpi == 1:
                subprocess.run('source %s %s;mpirun -np %s %s/bin/BAGEL %s/%s.json > %s/%s.log' % (self.mkl,self.arch,self.nproc,self.bagel,self.workdir,self.project,self.workdir,self.project),shell=True)
            else:
                subprocess.run('source %s %s;%s/bin/BAGEL %s/%s.json > %s/%s.log' % (self.mkl,self.arch,self.bagel,self.workdir,self.project,self.workdir,self.project),shell=True)
        os.chdir(maindir)

    def _read_bagel(self):
        ## read BAGEL logfile and pack data

        with open('%s/%s.log' % (self.workdir,self.project),'r') as out:
            log  = out.read().splitlines()

        ## pack energy, only includes the requested states by self.ci

        energy   = []
       	if os.path.exists('%s/ENERGY.out' % (self.workdir)) ==	True:
            energy   = np.loadtxt('%s/ENERGY.out' % (self.workdir))[0:self.ci]
        energy   = np.array(energy)

        ## pack force

        gradient = []
        for i in range(self.ci):
            if os.path.exists('%s/FORCE_%s.out' % (self.workdir,i)) == True:
                with open('%s/FORCE_%s.out' % (self.workdir,i)) as force:
                    g=force.read().splitlines()[1:self.natom+1]
                    g=S2F(g)
            else:
                g=[[0.,0.,0.] for x in range(self.natom)]

            gradient.append(g)
        gradient = np.array(gradient)

        ## pack NAC if requested
        ## if NAC is not requested, always clean nac and reshape to [1,self.natom,3]

        if self.read_nac == 1:
            coupling = []
            npair    = int(self.ci*(self.ci-1)/2)
            pairs    = NACpairs(self.ci).copy()
            for i in range(npair):
                pa,pb=pairs[i+1]
                if os.path.exists('%s/NACME_%s_%s.out' % (self.workdir,pa-1,pb-1)) == True:
                    with open('%s/NACME_%s_%s.out' % (self.workdir,pa-1,pb-1)) as nacme:
                        cp=nacme.read().splitlines()[1:self.natom+1]
                        cp=S2F(cp)
                else:
                    cp=[[0.,0.,0.] for x in range(self.natom)]
                coupling.append(cp)
            nac  = np.array(coupling) 
        else:
            nac  = np.zeros([1,self.natom,3])

        ## create zero np.array for civec and movec since they are not used
        civec    = np.zeros(0)
        movec    = np.zeros(0)

        return energy,gradient,nac,civec,movec

    def appendix(self,addons):
        ## fake function

        return self

    def evaluate(self,x):
        ## main function to run BAGEL calculation and communicate with other PyRAIMD modules

        ## setup BAGEL calculation
        self._setup_bagel(x)

        ## setup HPC settings
        if self.use_hpc == 1:
            self._setup_hpc()

        ## run BAGEL calculation
        self._run_bagel()

        ## read BAGEL output files
        energy,gradient,nac,civec,movec=self._read_bagel()

        ## clean up
        if self.keep_tmp == 0:
            shutil.rmtree(self.workdir)

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

