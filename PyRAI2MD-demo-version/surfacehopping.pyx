## Surface hopping for PyRAIMD
## Cython module
## Jingbai Li Sept 1 2020 

import sys
import numpy as np
cimport numpy as np

from tools import NACpairs

cdef avoid_singularity(float v_i,float v_j,int i,int j):
    ## This fuction avoid singularity of v_i-v_j for i < j 
    ## i < j assumes v_i <= v_j, thus assumes the sign of v_i-v_j is -1

    cdef float cutoff=1e-16
    cdef float sign, diff

    if i < j:
        sign=-1.0
    else:
        sign=1.0

    if   v_i == v_j:
        diff=sign*cutoff
    elif v_i != v_j and np.abs(v_i-v_j) < cutoff:
        diff=sign*cutoff
    elif v_i != v_j and np.abs(v_i-v_j) >= cutoff:
        #diff=v_i-v_j
        diff=sign*(v_i-v_j) # force v_i < v_j
    return diff

cdef Reflect(np.ndarray V,np.ndarray N,int reflect):
    ## This function refects velocity when frustrated hopping happens

    if   reflect == 1:
        V=-V
    elif reflect == 2:
        V-=2*np.sum(V*N)/np.sum(N*N)*N

    return V

cdef Adjust(float Ea, float Eb,np.ndarray V,np.ndarray M,np.ndarray N,int adjust,int reflect):
    ## This function adjust velocity when surface hopping detected
    ## This function call Reflect if frustrated hopping happens

    cdef float Ekin=np.sum(0.5*M*V**2)
    cdef frustrated=0
    cdef float dT,a,b,f

    if   adjust == 0:
        dT=Ea-Eb+Ekin
        if dT >= 0:
            f=1.0
        else:
            V=Reflect(V,N,reflect)
            frustrated=1

    elif adjust == 1:
        dT=Ea-Eb+Ekin
        if dT >= 0:
            f=(dT/Ekin)**0.5
            V=f*V
        else:
            V=Reflect(V,N,reflect)
            frustrated=1

    elif adjust == 2:
        a=np.sum(N*N/M)
        b=np.sum(V*N)
        dT=Ea-Eb
        dT=4*a*dT+b**2
        if dT >= 0:
           if b < 0:
               f=(b+dT**0.5)/(2*a)
           else:
               f=(b-dT**0.5)/(2*a)
           V-=f*N/M
        else:
            V=Reflect(V,N,reflect)
            frustrated=1

    return V,frustrated

cdef dPdt(np.ndarray A, np.ndarray H, np.ndarray D):
    ## This function calculate the gradient of state population
    ## The algorithm is based on Tully's method.John C. Tully, J. Chem. Phys. 93, 1061 (1990)

    ## State density A
    ## Hamiltonian H
    ## Non-adiabatic coupling D, D(i,j) = velo * nac(i,j)
    
    cdef int ci=len(A)
    cdef int k,j,l
    cdef np.ndarray dA=np.zeros((ci,ci),dtype=complex)

    for k in range(ci):
        for j in range(ci):
            for l in range(ci):
                dA[k,j]+=A[l,j]*(-1j*H[k,l]-D[k,l])-A[k,l]*(-1j*H[l,j]-D[l,j])

    return dA

cdef matB(np.ndarray A, np.ndarray H, np.ndarray D):
    ## This function calculate the B matrix
    ## The algorithm is based on Tully's method.John C. Tully, J. Chem. Phys. 93, 1061 (1990)

    ## State density A
    ## Hamiltonian H
    ## Non-adiabatic coupling D, D(i,j) = velo * nac(i,j)

    cdef int ci=len(A)
    cdef np.ndarray b=np.zeros((ci,ci))

    for k in range(ci):
        for j in range(ci):
            b[k,j]=2*np.imag(np.conj(A[k,j])*H[k,j])-2*np.real(np.conj(A[k,j])*D[k,j])

    return b

cpdef FSSH(dict traj):
    ## This function integrate the hopping posibility during a time step
    ## This function call dPdt to compute gradient of state population

    cdef np.ndarray A         = traj['A']
    cdef np.ndarray H         = traj['H']
    cdef np.ndarray D         = traj['D']
    cdef np.ndarray N         = traj['N']
    cdef int        substep   = traj['substep']
    cdef float      delt      = traj['delt']
    cdef int        iter      = traj['iter']
    cdef int        ci        = traj['ci']
    cdef int        state     = traj['state']
    cdef int        maxhop    = traj['maxh']
    cdef str        usedeco   = traj['deco']
    cdef int        adjust    = traj['adjust']
    cdef int        reflect   = traj['reflect']
    cdef int        verbose   = traj['verbose']
    cdef int        old_state = traj['state']
    cdef int        new_state = traj['state']
    cdef int        integrate = traj['integrate']
    cdef np.ndarray V         = traj['V']
    cdef np.ndarray M         = traj['M']
    cdef np.ndarray E         = traj['E']
    cdef float      Ekin      = traj['Ekin']

    cdef np.ndarray At=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray Ht=np.diag(E).astype(complex)
    cdef np.ndarray Dt=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray B=np.zeros((ci,ci))
    cdef np.ndarray dB=np.zeros((ci,ci))
    cdef np.ndarray dAdt=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray dHdt=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray dDdt=np.zeros((ci,ci),dtype=complex)

    cdef int n, i, j, k, p, stop, hoped, nhop, event, pairs, frustrated
    cdef float deco, z, gsum, Asum, Amm
    cdef np.ndarray Vt, g, tau, NAC
    cdef dict pairs_dict

    hoped=0
    n=0
    stop=0
    for i in range(ci):
        for j in range(i+1,ci):
            n+=1
            Dt[i,j]=np.sum(V*N[n-1])/avoid_singularity(E[i],E[j],i,j)
            Dt[j,i]=-Dt[i,j]

    if iter == 1:
        At[state-1,state-1]=1
        Vt=V
    else:
        dHdt=(Ht-H)/substep
        dDdt=(Dt-D)/substep
        nhop=0
        
        if verbose == 2:
            print('-------------- TEST ----------------')
            print('Iter: %s' % (iter))
            print('One step')
            print('dPdt')
            print(dPdt(A,H,D))
            print('matB')
            print(matB(A+dPdt(A,H,D)*delt*substep,H,D)*delt*substep)
            print('Integral')

        for i in range(substep):
            if integrate == 0:
                B=np.zeros((ci,ci))
            g=np.zeros(ci)
            event=0
            frustrated=0

            H+=dHdt
            D+=dDdt

            dAdt=dPdt(A,H,D)
            dAdt*=delt
            A+=dAdt
            dB=matB(A,H,D)
            B+=dB
            for p in range(ci):
                if np.real(A[p,p])>1 or np.real(A[p,p])<0:
                    A-=dAdt  # revert A
                    B-=dB
                    ## TODO p > 1 => p = 1 ; p<0 => p=0
                    stop=1   # stop if population exceed 1 or less than 0            
            if stop == 1:
                break

            for j in range(ci):
                if j != state-1:
                    g[j]+=np.amax([0,B[j,state-1]*delt/np.real(A[state-1,state-1])])

            z=np.random.uniform(0,1)

            gsum=0
            for j in range(ci):
                gsum+=g[j]
                nhop=np.abs(j+1-state)
                if gsum > z and nhop <= maxhop:
                    new_state=j+1
                    nhop=np.abs(j+1-state)
                    event=1
                    break

            if verbose >= 2:
                print('\nSubIter: %5d' % (i+1))
                print('NAC')
                print(Dt)
                print('D')
                print(D)
                print('A')
                print(A)
                print('B')
                print(B)
                print('Probabality')
                print(' '.join(['%12.8f' % (x) for x in g]))
                print('Population')
                print(' '.join(['%12.8f' % (np.real(x)) for x in np.diag(A)]))
                print('Random: %s' % (z))
                print('old state/new state: %s / %s' % (state, new_state))

            ## detect frustrated hopping and adjust velocity
            if event == 1:
                pairs_dict=NACpairs(ci)
                pairs=pairs_dict[str([state,new_state])]
                NAC=N[pairs-1] # pick up non-adiabatic coupling between state and new_state from the full array

                Vt,frustrated=Adjust(E[state-1],E[new_state-1],V,M,NAC,adjust,reflect)
                if frustrated == 0:
                    state=new_state

            ## decoherance of the propagation 
            if usedeco != 'OFF':
                deco=float(usedeco)
                tau=np.zeros(ci)

                ## matrix tau
                for k in range(ci):
                    if k != state-1:
                        tau[k]=np.abs(1/avoid_singularity(np.real(H[state-1,state-1]),np.real(H[k,k]),state-1,k))*(1+deco/Ekin) 

                ## update diagonal of A except for current state
                for k in range(ci):
                    for j in range(ci):
                        if k != state-1 and j != state-1:
                            A[k,j]*=np.exp(-delt/tau[k])*np.exp(-delt/tau[j])

                ## update diagonal of A for current state
                Asum=0.0
                for k in range(ci):
                    if k != state-1:
                        Asum+=np.real(A[k,k])
                Amm=np.real(A[state-1,state-1])
                A[state-1,state-1]=1-Asum

                ## update off-diagonal of A
                for k in range(ci):
                    for j in range(ci):
                        if   k == state-1 and j != state-1:
                            A[k,j]*=np.exp(-delt/tau[j])*(np.real(A[state-1,state-1])/Amm)**0.5
                        elif k != state-1 and j == state-1:
                            A[k,j]*=np.exp(-delt/tau[k])*(np.real(A[state-1,state-1])/Amm)**0.5

        ## final decision on velocity
        if state == old_state:   # not hoped
            Vt=V                 # revert scaled velocity
            hoped=0
        else:
            pairs_dict=NACpairs(ci)
            pairs=pairs_dict[str([old_state,state])]
            NAC=N[pairs-1] # pick up non-adiabatic coupling between state and new_state from the full array
            Vt,frustrated=Adjust(E[old_state-1],E[state-1],V,M,NAC,adjust,reflect)
            if frustrated == 0:  # hoped
                hoped=1
            else:                # frustrated hopping
                hoped=2

        At=A

    return At,Ht,Dt,Vt,hoped,old_state,state

cpdef GSH(dict traj):
    """ 
    Performs the calculation for global surface hopping proposed in Zhu-Nakamura Paper
    Created 10/04/2020 by Daniel Susman (susman.d)
    """
    cdef int        iter      = traj['iter']
    cdef int        ci        = traj['ci']
    cdef int        state     = traj['state']
    cdef int        old_state = traj['state']
    cdef int        adjust    = traj['adjust']
    cdef int        reflect   = traj['reflect']
    cdef int        verbose   = traj['verbose']
    cdef np.ndarray V         = traj['V']
    cdef np.ndarray M         = traj['M']
    cdef np.ndarray E         = traj['E']
    cdef np.ndarray Ep        = traj['Ep']
    cdef np.ndarray Epp       = traj['Epp']
    cdef np.ndarray G         = traj['G']
    cdef np.ndarray Gp        = traj['Gp']
    cdef np.ndarray Gpp       = traj['Gpp']
    cdef np.ndarray R         = traj['R']
    cdef np.ndarray Rp        = traj['Rp']
    cdef np.ndarray Rpp       = traj['Rpp']
    cdef float      Ekinp     = traj['Ekinp']
    cdef float      gap       = traj['gap']
    cdef int   	    test      = 0

    cdef int        i, hoped, frustrated, arg_min, arg_max
    cdef float      Etotp, dE, Ex, z
    cdef float      pi_over_four_term, b_in_denom_term, Psum
    cdef float      a_squared, b_squared, F_a, F_b, F_1, F_2
    cdef np.ndarray At,Ht,Dt,Vt,P,delE,NAC,N
    cdef np.ndarray f1_grad_manip_1,f1_grad_manip_2,f2_grad_manip_1,f2_grad_manip_2
    cdef np.ndarray begin_term, F_ia_1, F_ia_2,Mexp

    if iter > 2:
        z = np.random.uniform(0, 1) # random number
        Psum = 0                   # total probability
        P = np.zeros(ci)           # state hopping probablity (zeros vector using ci dimensions)

        # initialize a NAC matrix
        N = np.zeros([ci,V.shape[0],V.shape[1]])

        for i in range(ci):

            # determine the energy gap by taking absolute value
            delE = np.abs([E[i] - E[state - 1], Ep[i] - Ep[state - 1], Epp[i] - Epp[state - 1]])
        
            # total energy in the system at time t2 (t)
            Etotp = Ep[state - 1] + Ekinp

            # average energy in the system over time period
            Ex = (Ep[i] + Ep[state - 1]) / 2

            if np.argmin(delE) == 1 and np.abs(delE[1]) <= gap/27.211396132 and Etotp - Ex > 0:
                dE = delE[1]

                # Implementation of EQ 7
                begin_term = (-1 / (R - Rpp))
                if test == 1: print('EQ 7 R & Rpp: %s %s' % (R, Rpp))
                if test == 1: print('EQ 7 begin term: %s' % (begin_term))
                arg_min = np.argmin([i, state - 1])
                arg_max = np.argmax([i, state - 1])
                if test == 1: print('EQ 7 arg_max/min: %s %s' % (arg_max,arg_min))

                f1_grad_manip_1 = (G[arg_min]) * (Rp - Rpp)
                f1_grad_manip_2 = (Gpp[arg_max]) * (Rp - R)
                if test == 1: print('EQ 7 f1_1/f1_2: %s %s' % (f1_grad_manip_1,f1_grad_manip_1))

                F_ia_1 = begin_term * (f1_grad_manip_1 - f1_grad_manip_2)
                if test == 1: print('EQ 7 done: %s' % (F_ia_1))

                # Implementation of EQ 8
                f2_grad_manip_1 = (G[arg_max]) * (Rp - Rpp)
                f2_grad_manip_2 = (Gpp[arg_min]) * (Rp - R)
                F_ia_2 = begin_term * (f2_grad_manip_1 - f2_grad_manip_2)
                if test == 1: print('EQ 8 done: %s' % (F_ia_2))

                # Expand the dimesion of mass matrix M from [natom,1] to [3,natom,1] then do a transpotation
                # Reduce the dimesion of the transposed M from [1,natom,3] to [natom,3]
                Mexp = np.array([M, M, M]).T[0]

                # approximate nonadiabatic (vibronic) couplings, which are
                # left out in BO approximation
                NAC = (F_ia_2 - F_ia_1) / (M**0.5)
                NAC = NAC / (np.sum(NAC**2)**0.5)
                N[i]= NAC # add NAC to NAC matrix
                if test == 1: print('Approximate NAC done: %s' % (NAC))

                # EQ 4, EQ 5
                # F_A = ((F_ia_2 * F_ia_1)/mu)**0.5
                F_A = np.sum(NAC**2)**0.5
                if test == 1: print('EQ 4 done: %s' % (F_A))

                # F_B = (abs(F_ia_2 - F_ia_1) / mu**0.5)
                F_B = np.abs(np.sum((F_ia_2 * F_ia_1) / M))**0.5
                if test == 1: print('EQ 5 done: %s' % (F_B))

                # compute a**2 and b**2 from EQ 1 and EQ 2
                # ---- note: dE = 2Vx AND h_bar**2 = 1 in Hartree atomic unit
                a_squared = (F_A * F_B) / (2 * dE**3)
                b_squared = (Etotp - Ex) * (F_A / (F_B * dE))
                if test == 1: print('EQ 1 & 2 done: %s %s' % (a_squared,b_squared))

                # GOAL: determine sign in denominator of improved Landau Zener formula for switching 
                # probability valid up to the nonadiabtic transition region
                F_1 = E[i] - Epp[state - 1] # approximate slopes
                F_2 = E[state - 1] - Epp[i] # here

                if (F_1 == F_2):
                    sign = 1
                else:
                    # we know the sign of the slope will be negative if either F_1 or
                    # F_2 is negative but not the other positive if both positive or both negative
                    sign = np.sign(F_1 * F_2)
                if test == 1: print('Compute F sign done: %s' % (sign))

                # sign of slope determines computation of surface
                # hopping probability P (eq 3)
                pi_over_four_term = -(np.pi/ (4 *(a_squared)**0.5))
                if test == 1: print('P numerator done: %s' % (pi_over_four_term))
                b_in_denom_term = (2 / (b_squared + (np.abs(b_squared**2 + sign))**0.5))
                if test == 1: print('P denomerator done: %s' % (b_in_denom_term))
                P[i] = np.exp(pi_over_four_term * b_in_denom_term**0.5)
                if test == 1: print('P done: %s' % (P[i]))
            else:
                P[i] = 0

        # decide where to hop; ci = # states
        for i in range(ci):
            # sum up probability of each state until hit a random number, z
            # (see above)
            Psum += P[i]
            
            # describes when to stop accumulating the probability, based on stochastic point
            # find largest state w P > random number, z
            if Psum > z:
                state = i + 1 # surface hop has already happened, assign new state index
                hoped = 1   # has hop happened or not?
                break

    # if current state = old state, we know no surface hop has occurred
    if state == old_state:
        # Current velocity in this case will equal old velocity
        Vt = V

        # and mark this as no hop having occurred
        hoped = 0
    # Else, a hop has occurred
    else:
        # Velocity must be adjusted because hop has occurred
        Vt, frustrated = Adjust(E[old_state - 1], E[state - 1], V, M, N[state-1], adjust = 1, reflect = 2)

        # if frustrated is 0, we haven't had a frustrated hop
        if frustrated == 0:
            hoped = 1
        # else, we have a frustrated hop, which implies we must revert current state index to old index
        else:
            state = old_state
            hoped = 2

    # allocate zeros vector for population state density
    At = np.zeros([ci,ci])

    # assign state density at current state to 1
    At[state - 1, state - 1] = 1

    # Current energy matrix
    Ht = np.diag(E)

    # Current non-adiabatic matrix
    Dt = np.zeros([ci, ci])

    if iter > 2 and verbose >= 2:
        # prints taken from reference code
        print('Iter: %s' % (iter))
        print('Gap : %s' % (np.array([E[1]-E[0],Ep[1]-Ep[0],Epp[1]-Epp[0]])*27.211396132))
        print('E matrix')
        print(E)
        print('Ep matrix')
        print(Ep)
        print('Epp matrix')
        print(Epp)
        print('R matrix')
        print(R)
        print('Rp matrix')
        print(Rp)
        print('Rpp matrix')
        print(Rpp)
        print('G matrix')
        print(G)
        print('Gp matrix')
        print(Gp)
        print('Gpp matrix')
        print(Gpp)
        print('Random: %s' % (z))
        print('Probability')
        print(' '.join(['%12.8f' % (x) for x in P]))
        print('old state/new state: %s / %s' % (old_state, state))
        print('type: %s (nohop = 0, hoped = 1, frustrated = 2)' % (hoped))
        print('-----------------------------------------------------------')

    return At, Ht, Dt, Vt, hoped, old_state, state

cpdef NOSH(dict traj):
    """
    Fake surface hopping method to do single state molecular dynamics

    """
    cdef int        hoped     = 0
    cdef int        ci        = traj['ci']
    cdef int        state     = traj['state']
    cdef int        old_state = traj['state']
    cdef np.ndarray V         = traj['V']
    cdef np.ndarray E         = traj['E']

    # allocate zeros vector for population state density
    At = np.zeros([ci,ci])

    # assign state density at current state to 1
    At[state - 1, state - 1] = 1

    # Current energy matrix
    Ht = np.diag(E)

    # Current non-adiabatic matrix
    Dt = np.zeros([ci, ci])

    # Return the same velocity
    Vt = V

    return At, Ht, Dt, Vt, hoped, old_state, state

