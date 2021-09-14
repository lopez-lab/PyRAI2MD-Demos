## The Ab Inito Molecular Dynamics for PyQDynamics
## Jingbai Li Jun 9 2020

import time,datetime,os,pickle
import numpy as np
from reset_velocity import ResetVelo
from periodic_table import Element
from verlet import NoseHoover, VerletI, VerletII
from surfacehopping import FSSH,GSH,NOSH
from tools import Printcoord,NACpairs
class MIXAIMD:
    ## This class propagate nuclear position based on Velocity Verlet algorithm with hybrid QC/ML methods

    def __init__(self,variables_all,QM=None,REF=None,id=None,dir=None):
        ## T     : list 
        ##         Atom list
        ## R     : np.array
        ##         Coordinates in angstrom
        ## M     : np.array
        ##         Masses in amu
        ## V     : np.array
        ##         Velocity in Bohr/au
        ## Vs    : np.array 
        ##         Thermostat array
        ## E     : np.array
        ##         Energy in Eh
        ## G     : np.array
        ##         Gradient in Eh/Bohr
        ## N     : np.array
        ##         Non-adiabatic coupling in 1/Bohr
        ## M     : np.array
        ##         Nuclear mass in atomic unit
        ## t     : int
        ##         Dynamics time step
        ## maxh  : int
        ##         Maximum number of hoping between states
        ## delt  : float
        ##         Probability integration time step in atomic unit
        ## A     : np.array
        ##         Previous state denesity matrix
        ## H     : np.array
        ##         Previous energy matrix
        ## D     : np.array
        ##         Previous non-adiabatic matrix
        ## At    : np.array
        ##         Current state denesity matrix
        ## Ht    : np.array
        ##         Current energy matrix
        ## Dt    : np.array
        ##         Current non-adiabatic matrix
        ## Ekin  : float
        ##         Current kinetic energy
        ## state : int
        ##         Current state number
        ## deco  : float
        ##         Decoherance energy
        ## 1 au  = 2.4188843265857 * 10**-2 fs
        ## 1 kb  = 3.16881 * 10**-6 Eh/K

        self.timing = 0  ## I use this to test calculation time

        title          = variables_all['control']['title']

        self.fs_to_au=2.4188843265857*10**-2
        self.kb=3.16881*10**-6

        self.variables = variables_all
        self.version   = variables_all['version']
        self.maxerr_e  = variables_all['control']['maxenergy']
       	self.maxerr_g  = variables_all['control']['maxgradient']
       	self.maxerr_n  = variables_all['control']['maxnac']
        self.stop      = 0      ## stop aimd once error exceed maxerr
        self.traj      = variables_all['md'].copy()
        self.QM        = QM
        self.REF       = REF
        self.traj.update({
        'title'   : title,      ## name of calculation       
        'logpath' : os.getcwd(),## output directory
        'natom'   : 0,          ## number of atoms
        'A'       :np.zeros(0), ## previous state denesity matrix
        'H'       :np.zeros(0), ## previous energy matrix
        'D'       :np.zeros(0), ## previous non-adiabatic matrix
        'At'      :np.zeros(0), ## current state denesity matrix
        'Ht'      :np.zeros(0), ## current energy matrix
        'Dt'      :np.zeros(0), ## current non-adiabatic matrix
        'pciv'    : None,       ## previous ci vectors
        'pmov'    : None,       ## previous mo vectors
        'old'     : 0,          ## previous state number
        'state'   : 0,	        ## current state number
        'T'       :[],          ## atom list
        'V'       :np.zeros(0), ## velocity in Bohr/au
        'R'       :np.zeros(0), ## coordinates in angstrom
        'Rp'      :np.zeros(0), ## coordinates in the previous step
        'Rpp'     :np.zeros(0), ## coordinates in the previous previous step
        'Ekin'    : 0.,         ## kinetic energy in Eh
        'Ekinp'   : 0.,         ## kinetic energy in the previous step
        'Ekinpp'  : 0.,         ## kinetic energy in the previous previous step
        'E'       :np.zeros(0), ## potential energy in Eh
        'Ep'      :np.zeros(0), ## potential energy in the previous step
        'Epp'     :np.zeros(0), ## potential energy in the previous previous step
        'G'       :np.zeros(0), ## gradient in Eh/Bohr
        'Gp'      :np.zeros(0), ## gradient in the previous step
        'Gpp'     :np.zeros(0), ## gradient in the previous previous step
        'N'       :np.zeros(0), ## non-adiabatic coupling in 1/Bohr
        'Vs'      :np.zeros(0), ## thermostat array
        'iter'    : 0,          ## current iteration
        'hoped'   : 0,          ## surface hopping type
        'err_e'   : None,       ## error of energy in adaptive sampling
        'err_g'   : None,       ## error of gradient in adaptive sampling
        'err_n'   : None,       ## error of nac in adaptive sampling
        'MD_hist' :[],          ## md history
                         })

        self.choose_e      = ['qm','ref'][self.traj['ref_e']]
        self.choose_g  	   = ['qm','ref'][self.traj['ref_g']]
        self.choose_n  	   = ['qm','ref'][self.traj['ref_n']]
        self.traj['old']   = self.traj['root']
        self.traj['state'] = self.traj['root']
        self.record        = self.traj['record'] ## whether to record MD_hist
        self.direct        = self.traj['direct'] ## number of steps directly save output to disk
        self.buffer        = self.traj['buffer'] ## number of steps used for buffering output before saving to disk
        self.skipped       = 0                   ## number of steps skipped
        self.restart       = self.traj['restart']## turn on/off restart function
        self.addstep       = self.traj['addstep']## continue the trajectory with additional steps

        ###### obselete variables
        ## self.output_buffer = []                  ## list of buffered output
        ## self.energy_buffer = []	            ## list of buffered energy
        ## self.coord_buffer  = []  	       	    ## list of buffered coordinate

        ## update calculation title if the id is available
        if id != None:
            self.traj['title']  = '%s-%s' % (title,id)

       	## update calculation path if the directory name is available
        if dir != None:
            self.traj['logpath']= '%s/%s' % (os.getcwd(),self.traj['title'])
            if os.path.exists(self.traj['logpath']) == False:
                os.makedirs(self.traj['logpath'])

        ## check time step for microiteration in FSSH
        if self.traj['substep'] == 0:
            self.traj['delt']   = 0.2 
            self.traj['substep']= int(self.traj['size']/self.traj['delt'])
        else:
            self.traj['delt']   = self.traj['size']/self.traj['substep']

        ## check if it is a restart calculation and if the previous check point pkl file exists
        if self.restart == 1:
            check_log=os.path.exists('%s/%s.log' % (self.traj['logpath'],self.traj['title']))
            check_pkl=os.path.exists('%s/%s.pkl' % (self.traj['logpath'],self.traj['title']))
            if   check_log == True and check_pkl == True:
                with open('%s/%s.pkl' % (self.traj['logpath'],self.traj['title']),'rb') as mdinfo:
                    prevmd=pickle.load(mdinfo)
                self.traj.update(prevmd)
            elif check_log == True and check_pkl == False:
                print('\nCheckpoint file does not exist. Maybe you forgot to delete the old log file in a fresh calculation?')
                exit()
            elif check_log == False and check_pkl == True:
                print('\nPrevious log file does not exist. Maybe you forgot to delete the old checkpoint file in a fresh calculation?')
       	       	exit()

    def _propagate(self):
        # update previous-preivous and previous coordinates and kinetic energies
        self.traj['Rpp']   = self.traj['Rp']
        self.traj['Rp']    = self.traj['R']
        self.traj['Ekinpp']= self.traj['Ekinp']
        self.traj['Ekinp'] = self.traj['Ekin']

        # update current coordinates and kinetic energies
        self.traj['R'] = VerletI(self.traj)
        if self.timing == 1: print('verlet',time.time())
        xyz = self._write_coord(self.traj['T'],self.traj['R'])
        if self.timing == 1: print('write_xyz',time.time())
        self._compute_properties(xyz)
        if self.timing == 1: print('compute_egn',time.time())
        self.traj['V'] = VerletII(self.traj)
        if self.timing == 1: print('verlet_2',time.time())
        self.traj['Ekin'] = np.sum(0.5*(self.traj['M']*self.traj['V']**2))

        # reset velocity to avoid flying ice cube
        # end function early if velocity reset is not requested
        if self.traj['reset'] != 1:
            return None

        # end function early if velocity reset step is 0 but iteration is more than 1
        if self.traj['resetstep'] == 0 and self.traj['iter'] > 1:
            return None

        # end function early if velocity reset step is not 0 but iteration is not the multiple of it
        if self.traj['resetstep'] != 0:
            if self.traj['iter'] % self.traj['resetstep'] != 0:
                return None

        # finally reset velocity here
        V_noTR=ResetVelo(self.traj)
        self.traj['V']=V_noTR

    def _compute_properties(self,xyz):
        # update previous-previous and previous potential energies and forces
        self.traj['Epp']  = self.traj['Ep']
        self.traj['Ep']   = self.traj['E']
        self.traj['Gpp']  = self.traj['Gp']
        self.traj['Gp']   = self.traj['G']

        # update current potential energies and forces
        addons={
        'pciv' : self.traj['pciv'],
        'pmov' : self.traj['pmov'],
        }
        qm = self.QM
        qm.appendix(addons)

        ref = self.REF
        ref.appendix(addons)

        qm_results = qm.evaluate(xyz)
        ref_results = ref.evaluate(xyz)

        self.traj['pciv'] = ref_results['civec']
        self.traj['pmov'] = ref_results['movec']


        results={
                'qm'  : {
                        'energy'   : qm_results['energy'],
                        'gradient' : qm_results['gradient'],
                        'nac'      : qm_results['nac'],
                        'err_e'    : qm_results['err_e'],
                        'err_g'    : qm_results['err_g'],
                        'err_n'    : qm_results['err_n'],
                         },
                'ref' : {
       	       	       	'energy'   : ref_results['energy'],
                        'gradient' : ref_results['gradient'],
                        'nac'      : ref_results['nac'],
                        'err_e'    : ref_results['err_e'],
                        'err_g'    : ref_results['err_g'],
                        'err_n'    : ref_results['err_n'],
                         },
                 }

        self.traj['E']    = results[self.choose_e]['energy']
        self.traj['err_e']= results[self.choose_e]['err_e']
        self.traj['G']    = results[self.choose_g]['gradient']
        self.traj['err_g']= results[self.choose_g]['err_g']
        self.traj['N']    = results[self.choose_n]['nac']
        self.traj['err_n']= results[self.choose_n]['err_n']

        if self.record == 1:
            self.traj['MD_hist'].append([xyz,results[self.choose_e]['energy'].tolist(),results[self.choose_g]['gradient'].tolist(),results[self.choose_n]['nac'].tolist(),\
                                             results[self.choose_e]['err_e'],          results[self.choose_g]['err_g'],            results[self.choose_n]['err_n']]) # convert all to list

    def _thermostat(self):
        if self.traj['thermo'] == 1:
            V,Vs,Ekin = NoseHoover(self.traj)
            self.traj['V'] = V
            self.traj['Vs'] = Vs
            self.traj['Ekin'] = Ekin

    def _surfacehop(self):
        # update previous population, energy matrix, and non-adiabatic coupling matrix
        self.traj['A'] = np.copy(self.traj['At'])
        self.traj['H'] = np.copy(self.traj['Ht'])
        self.traj['D'] = np.copy(self.traj['Dt'])

        # update current population, energy matrix, and non-adiabatic coupling matrix
        if   self.traj['sfhp'] == 'fssh':
            At,Ht,Dt,V,hoped,old_state,state=FSSH(self.traj)
        elif self.traj['sfhp'] == 'gsh':
            At,Ht,Dt,V,hoped,old_state,state=GSH(self.traj)
        elif self.traj['sfhp'] == 'nosh':
            At,Ht,Dt,V,hoped,old_state,state=NOSH(self.traj)

        self.traj['At'] = At
        self.traj['Ht'] = Ht
        self.traj['Dt'] = Dt
        self.traj['V']  = V
        self.traj['hoped'] = hoped
        self.traj['old']   = old_state
        self.traj['state'] = state

        if self.record == 1:
            self.traj['MD_hist'][-1].append(np.diag(np.real(At)).tolist())

    def _read_coord(self,xyz):
        xyz = np.array(xyz)
        natom = len(xyz)
        T = xyz[:,0]
        R = xyz[:,1:].astype(float)
        M = np.array([Element(x).getMass()*1822.8852 for x in T]).reshape([-1,1])
        return T,R,M

    def _write_coord(self,T,R):
        xyz = []
        for n,i in enumerate(R):
            exyz = [T[n]] + i.tolist()
            xyz.append(exyz)
        return xyz

    def _heading(self):

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |          Nonadiabatic Molecular Dynamics          |
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

    def	_chkpoint(self):
        ## This function print current information
        ## This function append output to .log, .md.energies and .md.xyz

        Chk       = self.traj.copy()          ## copy the dict in case I will change the data type for saving in the future
        title     = Chk['title']              ## title
        logpath   = Chk['logpath']            ## output directory
        temp      = Chk['temp']               ## temperature
        t         = Chk['size']               ## time step size
        ci        = Chk['ci']                 ## ci dimension
        old_state = Chk['old']                ## the previous state or the current state before surface hopping
        state     = Chk['state']              ## the current state or the new state after surface hopping
        iter      = Chk['iter']               ## the current iteration
        T         = Chk['T'].reshape([-1,1])  ## atom list
        R         = Chk['R']                  ## coordiantes
        V         = Chk['V']                  ## velocity
        Ekin      = Chk['Ekin']               ## kinetic energy
        E         = Chk['E']                  ## potential energy
        G         = Chk['G']                  ## gradient
        N         = Chk['N']                  ## non-adiabatic coupling
        At        = Chk['At']                 ## population (complex array)
        hoped     = Chk['hoped']              ## surface hopping detector
        natom     = len(T)                    ## number of atoms
        err_e     = Chk['err_e']              ## error of energy in adaptive sampling
        err_g     = Chk['err_g']              ## error of gradient in adaptive sampling
        err_n     = Chk['err_n']              ## error of nac in adaptive sampling
        verbose   = Chk['verbose']            ## print level

        ## prepare a comment line for xyz file
        cmmt='%s coord %d state %d' % (title,iter,old_state)

        ## prepare the surface hopping detection section according to Molcas output format
        if   hoped == 0:
            hop_info=' A surface hopping is not allowed\n  **\n At state: %3d\n' % (state)
        elif hoped == 1:
       	    hop_info=' A surface hopping event happened\n  **\n From state: %3d to state: %3d *\n' % (old_state,state)
            cmmt+=' to %d CI' % (state)
        elif hoped == 2:
            hop_info=' A surface hopping is frustrated\n  **\n At state: %3d\n' % (state)

        ## prepare population and potential energy info
        pop=' '.join(['%28.16f' % (x) for x in np.real(np.diag(At))])
        pot=' '.join(['%28.16f' % (x) for x in E])

        ## prepare non-adiabatic coupling pairs
        pairs=NACpairs(ci)

        ## start to output
        log_info=' Iter: %8d  Ekin = %28.16f au T = %8.2f K dt = %10d CI: %3d\n Root chosen for geometry opt %3d\n' % (iter,Ekin,temp,t,ci,old_state)
        log_info+='\n Gnuplot: %s %s %28.16f\n  **\n  **\n  **\n%s\n' % (pop,pot,E[old_state-1],hop_info)

        if verbose >= 1:
            xyz=np.concatenate((T,R),axis=1)
            log_info+="""
  &coordinates in Angstrom
-------------------------------------------------------
%s-------------------------------------------------------
""" % (Printcoord(xyz))
            velo=np.concatenate((T,V),axis=1)
            log_info+="""
  &velocities in Bohr/au
-------------------------------------------------------
%s-------------------------------------------------------
""" % (Printcoord(velo))
            for n,g in enumerate(G):
                grad=np.concatenate((T,g),axis=1)
                log_info+="""
  &gradient %3d in Eh/Bohr
-------------------------------------------------------
%s-------------------------------------------------------
""" % (n+1,Printcoord(grad))
            for m,n in enumerate(N):
                nac=np.concatenate((T,n),axis=1)
       	        log_info+="""
  &non-adiabatic coupling %3d - %3d in 1/Bohr
-------------------------------------------------------
%s-------------------------------------------------------
""" % (pairs[m+1][0],pairs[m+1][1],Printcoord(nac))

        if err_e != None and err_g != None and err_n != None:
            log_info+="""
  &error iter %-10s
-------------------------------------------------------
  Energy   StDev:             %-10.4f
  Gradient StDev:             %-10.4f
  Nac      StDev:             %-10.4f
-------------------------------------------------------

""" % (iter,err_e,err_g,err_n)
       	    if err_e > self.maxerr_e or err_g > self.maxerr_g or err_n > self.maxerr_n:
                self.stop = 1

        energy_info='%8.2f%28.16f%28.16f%28.16f%s\n' % (iter*t,E[old_state-1],Ekin,E[old_state-1]+Ekin,pot)
        xyz_info='%d\n%s\n%s' % (natom,cmmt,Printcoord(np.concatenate((T,R),axis=1)))

        if Chk['silent'] == 0:
            print(log_info)

        #print(log_info)
        self._dump_to_disk(Chk,logpath,title,log_info,energy_info,xyz_info)

######### I do not use the code below because I don't want to keep the huge data in RAM
#        if   self.traj['iter'] <= self.direct:
#            self._dump_to_disk(logpath,title,log_info,energy_info,xyz_info)
#        elif self.traj['iter'] > self.direct:
#            self.output_buffer.append(log_info)
#            self.energy_buffer.append(energy_info)
#            self.coord_buffer.append(xyz_info)
#            if  len(self.output_buffer) == self.buffer or self.traj['iter'] == self.traj['step'] or self.stop == 1:
#                log_info=''.join(self.output_buffer)
#                energy_info=''.join(self.energy_buffer)
#                xyz_info=''.join(self.coord_buffer)
#                self._dump_to_disk(logpath,title,log_info,energy_info,xyz_info)
#                self.output_buffer=[]

    def _dump_to_disk(self,chk,logpath,title,log_info,energy_info,xyz_info):
        ## serialize the md calculation info for restart
        if self.restart == 1:
            with open('%s.pkl' % (title),'wb') as mdinfo:
                pickle.dump(chk,mdinfo)

        ## output data to disk
        mdlog=open('%s/%s.log' % (logpath,title),'a')
        mdlog.write(log_info)
        mdlog.close()

        mdenergy=open('%s/%s.md.energies' % (logpath,title),'a')
        mdenergy.write(energy_info)
        mdenergy.close()

        mdxyz=open('%s/%s.md.xyz' % (logpath,title),'a')
        mdxyz.write(xyz_info)
        mdxyz.close()

    def run(self,xyz,velo):
        ## xyz  : list
        ##        Coordinates list of [atom x y z] in angstrom
        ## velo : np.array
        ##        Nuclear velocities in Bohr/au

        title    = self.traj['title']
        logpath  = self.traj['logpath']
        warning  = ''

        start=time.time()
        heading='Nonadiabatic Molecular Dynamics Start: %20s\n%s' % (self._whatistime(),self._heading())

        if self.traj['silent'] == 0:
            print(heading)

        if self.restart == 0 or os.path.exists('%s/%s.log' % (logpath,title)) == False:
            ## write new log when it does not exist or it is a fresh md calculation
            ## otherwise, the new results will be appended to the existing log in a restart calculation

            mdlog=open('%s/%s.log' % (logpath,title),'w')
            mdlog.write(heading)
            mdlog.close()

            mdhead='%8s%28s%28s%28s%28s\n' % ('time','Epot','Ekin','Etot','Epot1,2,3...')
            mdenergy=open('%s/%s.md.energies' % (logpath,title),'w')
            mdenergy.write(mdhead)
            mdenergy.close()

            mdxyz=open('%s/%s.md.xyz' % (logpath,title),'w')
            mdxyz.close()

            natom = len(xyz)
            T,R,M = self._read_coord(xyz)

            self.traj['natom'] = natom
            self.traj['T'] = T
            self.traj['R'] = R
            self.traj['M'] = M
            self.traj['V'] = velo

        completed=self.traj['iter']
        self.traj['step']+=self.addstep
        for iter in range(self.traj['step']-completed):
            self.traj['iter'] = iter+1+completed
            if self.timing == 1: print('start', time.time())
            self._propagate()    # update E,G,N,R,V,Ekin
            if self.timing == 1: print('propagate',time.time())
            self._thermostat()   # update Ekin,V,Vs
            if self.timing == 1: print('thermostat',time.time())
            self._surfacehop()   # update A,H,D,V,state
            if self.timing == 1: print('surfacehop',time.time())

            if   self.traj['iter'] <= self.direct:
                self._chkpoint()
            else:
                self.skipped+=1
                if  self.skipped == self.buffer or self.traj['iter'] == self.traj['step'] or self.stop == 1:
                    self._chkpoint()
                    self.skipped = 0

            if self.timing == 1: print('save',time.time())
            if self.stop == 1:
#                if len(self.traj['MD_hist']) > 1:
#                    self.traj['MD_hist'] = self.traj['MD_hist'][:-1] # revert one step back if trajectory has more than one step, since the large error
                warning='Errors are too large'
                break


        end=time.time()
        walltime=self._howlong(start,end)
        tailing='%s\nNonadiabatic Molecular Dynamics End: %20s Total: %20s\n' % (warning,self._whatistime(),walltime)

        if self.traj['silent'] == 0:
            print(tailing)

        mdlog=open('%s/%s.log' % (logpath,title),'a')
        mdlog.write(tailing)
        mdlog.close()

        return self.traj['MD_hist']


