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

- **/[2+2]-photocycloaddition_toward_Cubane/TOD-8Me** Octamethyl [3]-ladderene model.

- **/[2+2]-photocycloaddition_toward_Cubane/TOD-8CF3** Octatrifluoromethyl [3]-ladderene model.

- **/[2+2]-photocycloaddition_toward_Cubane/TOD-8pr** Octacyclopropyl [3]-ladderene model.

- **json2txt.py** This is a script to conver the training data from JSON to human-readable TXT file. 

Each folder contains (e.g., in TOD-8Me):

- **NN-tod-8me/** This folder contains the trained NN weights and hyperparameters in the /energy_gradient/ subfolder.

- **training_data/** This folder contains the compressed training data, saved separatly in a maximum 10 MB file.

- **tod-8me-1/** This folder contains the example calculation.

- **tod-8me.init.tar.xz** This is the compressed initial conditions sampled by Wigner sampling at zero-point energy level.

- **allpath** This file stores the permutation map used for training NN.

- **invd_index3.txt** This file stores the atom indices defining the inverse distance

  training data, Wigner sampled initial conditions, and a calculation example. 

# How to use this demo?
## Extracting trainding data and initial conditions
Download the repository

    git clone https://github.com/lopez-lab/PyRAI2MD-Demos.git
    
Copy one of the model folder or go to that folder (e.g., TOD-8Me)

    cd /[2+2]-photocycloaddition_toward_Cubane/TOD-8Me

To extract the training data first combine the individule files then untar it

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



## Getting familiar with the PyRAI2MD input file
[under constructinon]
## Understanding the inverse distance file and permutation map
[under constructinon]
# What to expect from the simulation?
## Anatomy of logfiles
[under constructinon]
## Additional notes
[under constructinon]
