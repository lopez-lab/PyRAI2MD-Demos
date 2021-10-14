# Python Rapid Artificial Intelligence Ab Initio Molecular Dynamics Demo
This is a demostration of running machine-learning nonadiabatic molecular dyanmics (ML-NAMD) simulation using PyRAI2MD.
These ML-NAMD simulations were used to investigate the substituent effects on the [2+2]-photocycloaddition of [3]-ladderenes
toward a class of energy-dense molecule, cubane.(ChemRxiv *preprint*, doi:10.33774/chemrxiv-2021-lxsjk).
## Prerequisite
 - **Python >=3.7** PyRAI2MD is written and tested in Python 3.7.4. Older version of Python is not tested and might not be working properly.
 - **TensorFlow >=2.2** TensorFlow/Keras API is required to load the trained NN models and predict energy and force.
 - **Cython** PyRAI2MD uses Cython library for efficient surface hopping calculation.
 - **Matplotlib/Numpy** Scientifc graphing and numerical library for plotting training statistic and array manipulation.
 
# What does this demo contain?
## PyRAI2MD demo version
**/PyRAI2MD-demo-version/**

A demo version of PyRAI2MD for testing ML-NAMD simulation. This version was only used for substituted [3]-ladderenes. Some old functionalities were obselete or rebuilt in current develop version. There is no warranty that this version will work properly for other molecules.

**/PyRAI2MD-demo-version/pyNNsMD/**

The ML kernel of PyRAI2MD developed by Patrick Reiser and Pascal Friederich @KIT, Germany. This is an outdated version that was initially interfaced with this demo version of PyRAI2MD. The latest release is here (https://github.com/aimat-lab/NNsForMD).

## Trained NN models
This demo has three trained NN models:

- **/Photocycloaddition_toward_Cubane/TOD-8Me** Octamethyl [3]-ladderene model.

- **/Photocycloaddition_toward_Cubane/TOD-8CF3** Octatrifluoromethyl [3]-ladderene model.

- **/Photocycloaddition_toward_Cubane/TOD-8pr** Octacyclopropyl [3]-ladderene model.

- **json2txt.py** This is a script to conver the training data from JSON to human-readable TXT file. 

Each folder contains (e.g., in TOD-8Me):

- **NN-tod-8me/energy_gradient/** This folder contains the NN hyperparameters, training logs and trained weights.

- **training_data/** This folder contains the compressed training data, saved separatly in a maximum 10 MB file.

- **tod-8me-1/** This folder contains the example calculation.

- **tod-8me.init.tar.xz** This is the compressed initial conditions sampled by Wigner sampling at zero-point energy level.

- **allpath2** This file stores the permutation map used for training NN. Note each demo has a different permuation map.

- **invd_index3.txt** This file stores the atom indices defining the inverse distance

  training data, Wigner sampled initial conditions, and a calculation example. 

# How to use this demo?
## Extracting trainding data and initial conditions
Download the repository

    git clone https://github.com/lopez-lab/PyRAI2MD-Demos.git
    
Copy one of the model folder or go to that folder (e.g., TOD-8Me)

    cd ./PyRAI2MD-Demos/Photocycloaddition_toward_Cubane/TOD-8Me

To extract the training data, first combine the individule files then untar it

    cd training_data
    cat data9303-13.json.tar.xz.part** > data9303-13.json.tar.xz
    tar -xvf data9303-13.json.tar.xz

The training data are saved in a capatable dictionary that PyRAI2MD can directly read. It can be converted to a more human-readable TXT file by

    python convert.py data9303-13.json

To load the training data in Python

    import json
    with open('data9303-13.json','r') as indata
         data = json.load(indata)
         natom, nstate, xyzset, invrset, energyset, gradset, nacset, civecset, movecset = idata
         
    """
    natom             Number of atoms.
    
    nstate            Number of state.
    
    xyzset            Nuclear coordinates in a N by natom by 4 list.\
                      N is the number of data points. 4 is the dimension of XYZ format (e.g., [atom, X, Y, Z]).
    
    energyset         Electronic energy in a N by nstate list. 
                      N is the number of data points.
                      
    gradset           Nuclear gradient coordinates in N by nstate by natom by 3 list.
                      N is the number of data points. 3 is the dimension of the gradient (e.g., [X, Y, X]).
    
    nacset            Nonadiabatic coupling (numerator part) in N by nstate*(nstate-1)/2 by natom by 3 list.
                      N is the number of data points. 3 is the dimension of the coupling (e.g., [X, Y, X]).
                      Here it only stores the unique interstate coupling (i.e., upper triangle of the coupling matrix).
                      Since the nonadibatic coupling was not used, they are all zero matrix.
    
    civecset          Addition list to store CI vector information, not used in this cases, thus are empty.
    
    movecset          Addition list to store MO vector information, not used in this cases, thus are empty.
    """

To extract the initial condition, simply untar it

     tar -xvf tod-8me.init.tar.xz
     
     """
     The initial conditions are save in a special XYZ format, look like this:
     Init   1  40   X(A)     Y(A)     Z(A)     VX(au)     VY(au)     VZ(au)     g/mol    e
     C            0.0000   0.0000    0.000     0.0000     0.0000     0.0000    12.011    6
     ...
     ...
     Init   2  40   X(A)     Y(A)     Z(A)     VX(au)     VY(au)     VZ(au)     g/mol    e
     ...
     ...
     
     Each structure start with a title line begining with "Init" and followed by 
     the trajectory index and the number of atom.
     
     The following "X(A) ... VZ(au)" are just notes for those columns.
     
     The next line specify the atom type, nuclear coordinates in X, Y, and Z, 
     velocities in X, Y, Z, molar mass, and nuclear charges 
     
     The nuclear coordinates are in Angstrom. 
     The velocities are in atomic unit, Bohr/atomic unit of time
     """
     
## Getting familiar with the PyRAI2MD
To get started with PyRAI2MD, go to the example calculation folder

    cd ./tod-8me-1
    
PyRAI2MD requires an input file, a nuclear coordiantes file and a velocities file to start ML-NAMD simulation.    

The input file has been pre-configured. Here, we list some of the important keywords.

Under &CONTROL section

    title             This is the title of calculation. 
                      It assumes the NN model folder, nuclear coordiantes, and velocities have the same basename, e.g., 
                          the NN model folder should be NN-$title,
                          the nuclear coordiantes file should be $title.xyz, and 
                          the velocities file should be $title.velo.

    maxenergy         The energy threshold to terminated a trajectories if the prediction std exceeds the treshold.
    
    maxgradient       The gradient threshold to terminated a trajectories if the prediction std exceeds the treshold.

Under &MD section

    initcond          The option to sample initial condition.
                      0 means reading from external coordinates file.
                      As such, the related keywords, nesmb, method, format are disabled.

    step              The number of trajectory steps.
    
    size              The stepsize of trajectory.
    
    ci                The configurational interaction space dimension.
                      This is equivalent to the number of states.
                      
    root              The starting root counting from 1 for the ground-state.

    sfhp              The surface hopping method.
                      The option gsh uses the Zhu-Nakamura approximation.
    
    gap               The energy gap to detect surface hopping using Zhu-Nakamura Approximation. Only work with the gsh option.
    
    
    thermo            The thermodynamic ensemble option.
                      0 means microconanical ensemble, NVE.

    silent            The screen printing option.
                      1 turns off the screen printing process to speed up ML-NAMD simulation, 
                      otherwise it prints logfile on screen every MD step.
                      
    verbose           The output printing option.
                      0 only writes energies, population and surface hopping information,
                      which speed up ML-NAMD simulation and reduce the size of logfile.
                      1 writes nuclear coordinates, velocities, and nonadiabatic couplings
                      to disk every MD step.
     
Under &NN section

    modeldir          The path to the NN model folder (current folder or absolute path).

    train_data        The path to the training data (current folder or absolute path).
    
    silent            The NN writing option.
                      1 turns off the NN writing process to speed up ML-NAMD simulation,
                      otherwise, it writes the prediction to disk every MD step.

    nn_eg_type        The type of energy_gradient model.
                      2 loads two different NNs to predict energy and gradient together
                      
    nn_nac_type       The type of nonadibatic_coupling model
                      0 skips the nonadibatic_coupling NN.
                      
    permute_map       The path to the permutation map file (current folder or absolute path).
    
Under &EG section (the same for &EG2 section)

    invd_index         The path to the inverse distance file (current folder or absolute path).
                       Skip this keyword will use the full inverse distance matrix.

    depth              Number of hidden layers.
   
    nn_size            Number of neurons per hidden layer.
    
    loss_weights       The weight between energy loss and gradient loss.
    
    val_split          The ratio of validation set.
    
    
The nuclear coordinates file use the simple XYZ format as:

    40
    comment line
    C       0.000  0.000  0.000
    C       0.001  0.000  1.450
    ...

The velocities file only contains the XYZ component as:

    0.000 0.000 0.000
    0.001 0.000 1.450
    ....
    
    
## Running the first PyRAI2MD calculation
PRAI2MD calculation is easy to start. In only requires one enviromental viriable "PYRAIMD" pointing to the location of PyRAI2MD-demo-version.

    export PYRAIMD = /location/to/PyRAI2MD-demo-version
    python $PYRAIMD/PYRAIMD.py input

One can use the run_PyRAIMD.sh to submit the job to SLURM, or simply use the inside command to run it locally.

## Understanding the inverse distance file and permutation map
The inverse distance and permutation map are important to train the NNs, but not affect the ML-NAMD simulations. Here we just briefly introduce how they are working with PyRAI2MD. The theoretial background of permutation and inverse distance will not be explained here, but available in our publications.

The inverse distance file, **invr_dist3.txt**, is a text file looks like this:

    1 2
    1 3
    1 4
    ...
    ...

Each line contains the indices of two atom defining a distance to computer their inverse and gradients of inverse.
By removing some distances, such as a long distance outside a given radius with respect to an atom, it reduces the NN input size. As most of the long distance results in a nearly zero inverse, removing them help remove the noise in the input features.

The permutation map file, **allpath2**, is a text file looks like this:

    5 3 2 8 1 7 6 4 ....

Each line specifies one permutation of the orginal atom indices. The meanning is

    1 permute with 5
    2 permute with 3
    3 permute with 2
    4 permute with 8
    5 permute with 1
    6 permute with 7
    7 permute with 6
    8 permute with 4

# What to expect from the simulation?
## Anatomy of logfiles
Be prepared that the ML-NAMD simulation will finish very soon and generate a huge amount of data!
It is always laborious to search many useful MD informtion from a long and redundant trajectory logfile. PyRAI2MD outputs the trajectory data into several files to enable convenient trajectory analysis.

The output files contain:

- **tod-8me-1.log**  A general output that includes all input settings and MD data, such as kinetic energy, potential energy, surface hopping events and state populations.

- **tod-8me-1.md.energies**  A text file only stores the kinetic energy and potential energy. 

- **tod-8me-1.md.xyz**       A text file only stores the nuclear coordiantes. The comment line record the MD step and electronic state.

- **tod-8me-1.md.velo**      A text file only store the velocities. The comment line record the MD step and electronic state.

- **tod-8me-1.sh.energies**  A text file only stores the kinetic energy and potential energy at surface hopping points. 

- **tod-8me-1.sh.xyz**       A text file only stores the nuclear coordiantes at surface hopping points.

- **tod-8me-1.sh.velo**      A text file only store the velocities at surface hopping points.

## Additional notes
TBD
