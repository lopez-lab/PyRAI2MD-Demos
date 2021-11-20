## verlocity verlet for PyRAIMD
## Jingbai Li Feb 13 2020

import numpy as np

def NoEnsemble(traj):
    ## This function does not adjust energy and velocity
    V        = traj['V']
    Ekin     = traj['Ekin']
    Vs = [-1,0,0,0,-1] # Ignore the first 4 element since NVE do not use them
                       # Set the first element to -1 to reset the NoseHoover if necessary

    return V,Vs,Ekin

def NVE(traj):
    ## This function rescale velocity as NVE ensemble
    iter     = traj['iter']
    V        = traj['V']
    Ekin     = traj['Ekin']
    Vs       = traj['Vs']   # here I borrow Vs as total energy
    E        = traj['E']
    state    = traj['state']

    if iter > 1 and Vs[4] == -1:
       	reset =	1
    else:
       	reset =	0

    if iter == 1 or reset == 1:
        Vs = [-1,0,0,0,E[state-1]+Ekin] # Ignore the first 4 element since NVE do not use them
                                        # Set the first element to -1 to reset the NoseHoover if necessary
        K  = Ekin
    else:       
        K  = Vs[4]-E[state-1]  # Vs[4] should be always larger than current state energy
        if K <=0:              # if the Vs[4] become negative due to energy prediction error
            K=1e-8             # damp kinetic energy to 1e-8 a.u.
        V *= (K/Ekin)**0.5

    return V,Vs,K

def NoseHoover(traj):
    ## This function calculate velocity scale factor by Nose Hoover thermo stat from t to t/2

    iter     = traj['iter']
    natom    = traj['natom']
    V        = traj['V']
    Ekin     = traj['Ekin']
    Vs       = traj['Vs']
    temp     = traj['temp']
    t        = traj['size']
    kb       = 3.16881*10**-6
    fs_to_au = 2.4188843265857*10**-2

    if iter > 1 and Vs[0] == -1:
        reset = 1
    else:
        reset = 0

    if iter == 1 or reset == 1:
        freq=1/(22/fs_to_au) ## 22 fs to au Hz
        Q1=3*natom*temp*kb/freq**2
        Q2=temp*kb/freq**2
        Vs=[Q1,Q2,0,0,-1]     ## The original Vs matrix only has Q1,Q2,V1,and V2.
                              ## Here the 5th element is for total energy, which will not be use in NoseHoover but NVE.
                              ## Set the 5th element to -1 to reset NVE if necessary
    else:
        Q1,Q2,V1,V2,_=Vs     ## Ignore the total energy (the 5th element).
        G2=(Q1*V1**2-temp*kb)/Q2
        V2+=G2*t/4
        V1*=np.exp(-V2*t/8)
        G1=(2*Ekin-3*natom*temp*kb)/Q1
        V1+=G1*t/4
        V1*=np.exp(-V2*t/8)
        s=np.exp(-V1*t/2)

        Ekin*=s**2

        V1*=np.exp(-V2*t/8)
        G1=(2*Ekin-3*natom*temp*kb)/Q1
        V1+=G1*t/4
        V1*=np.exp(-V2*t/8)
        G2=(Q1*V1**2-temp*kb)/Q2
        V2+=G2*t/4
        Vs[2]=V1
        Vs[3]=V2
        V*=s

    return V,Vs,Ekin

def VerletI(traj):
    ## This function update nuclear position
    ## R in Angstrom, 1 Bohr = 0.529177249 Angstrom
    ## V in Bohr/au
    ## G in Eh/Bohr
    ## M in atomic unit

    iter  = traj['iter']
    R     = traj['R']
    V     = traj['V']
    G     = traj['G']
    M  	  = traj['M']
    t     = traj['size']
    state = traj['state']
    GD    = traj['graddesc']

    if GD == 1:
        V = np.zeros(V.shape)

    if iter > 1:
        G = G[state-1]
        R+= (V*t-0.5*G/M*t**2)*0.529177249
    return R

def VerletII(traj):
    ## This function update velocity

    iter  = traj['iter']
    M     = traj['M']
    G     = traj['G']
    G0    = traj['Gp']
    V     = traj['V']
    t     = traj['size']
    state = traj['state']
    GD    = traj['graddesc']

    if iter > 1:
        G0= G0[state-1]
        G = G[state-1]
        V-= 0.5*(G0+G)/M*t

    if GD == 1:
       	V = np.zeros(V.shape)
    return V

