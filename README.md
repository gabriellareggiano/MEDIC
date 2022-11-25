# MEDIC: Model error detection in cryo-EM

MEDIC is a statistical model derived from logistic regression that will identify possible errors in your structure. It will predict a probability of error (high value = more likely to be an error) for every residue.

## Manuscripts

Robust residue-level error detection in cryo-electron microscopy models. Gabriella Reggiano, Daniel Farrell, Frank DiMaio (https://www.biorxiv.org/content/10.1101/2022.09.12.507680v1)

## Installation/dependencies

To install MEDIC, git clone the source code and install the following with anaconda

NEED INSTRUCTIONS FOR PYROSETTA INSTALL

- Apply  for a license (free for academic use) [here](https://els2.comotion.uw.edu/product/rosetta)
- Add the PyRosetta channel to your *~/.condarc* and replace the *username* and *password* with your information
```
    channels:
        - https://username:password@conda.graylab.jhu.edu
        - defaults
```

- Create a conda environment for MEDIC (here called *medic*):

```
    conda create -n medic -y python=3 pytorch pyrosetta
    conda install -n medic -y tensorflow keras requests more-itertools matplotlib
    conda install -n medic -y click=7.1.2 dask=2.30 dask-jobqueue
    conda activate medic
    pip install mmtf-python sklearn
```

- Install MEDIC into the active conda environment
```
    conda activate medic
    git clone --recursive https://github.com/gabriellareggiano/private_MEDIC.git
    cd private_MEDIC
    git submodule set-url DeepAccNet https://github.com/gabriellareggiano/DeepAccNet.git
    git submodule update --init
    python setup.py install
```


## Running MEDIC on your structure
### MEDIC background

MEDIC has four steps, all performed with this script:
1. Local relax in Rosetta [^1]
2. Calculation of density z-scores
3. Calculation of predicted lDDTs [^2]
4. Error prediction

### MEDIC command

You can run the following to see all options for MEDIC
```
./path/to/MEDIC/detect_errors.py --help
```

The minimal command is shown below. Make sure your pdb is docked into the map before running.
```
./path/to/MEDIC/detect_errors.py –pdb {path/to/pdb} –map {path/to/map} –reso {global resolution} –j {number_processes}
```
If your structure has already been relaxed with Rosetta, add the flag: —skip_relax
  - The relax is mandatory, don’t skip if your pdb hasn’t been through Rosetta

If your structure has ligands or nucleac acids or noncanonical amino acids, add the flag:  –clean 

## Visualizing/inspecting outputs

### MEDIC outputs the following files:

- {pdb}_refined.pdb
  - this is your structure after the local relax

- {pdb}_MEDIC_bfac_pred.pdb
  - this is your relaxed structure with the probabilities in the B-factor column

- {pdb}_MEDIC_predictions.csv
  - this contains all the relevant scores for every residue with the predicted probabilities

- MEDIC_summary.txt
  - this contains all the segments that have been marked as errors, as well as the scores that flagged them as errors
```
              25R - 28R, definite error       —> high probability error, residues 25-28, chain R
                   causes: density               —> density score alone can predict this to be an error

              23R - 24R, possible error     —> low probability error, residues 23-24, chain R
                   causes: density + lddt     —> the density and the lddt together predict an error
```

### To view your error predictions in Chimera:

1. Load in your *bfac_pred.pdb to Chimera. 
2. Go to Tools -> Depiction -> Render by attribute
3. In the Render by attribute window:
   - Attributes of -> residues
   - Attribute -> average -> bfactor
   - Choose your thresholds for coloring we recommend:
     - 0.78 -> far right histogram
     - 0.60 -> left
   - Note that setting this value to lower than 0.60 may allow you to find more errors, but MEDIC will also mark more false positives.


### To view your error predictions in ChimeraX:

1. Load in your *bfac_pred.pdb to Chimera
2. Enter the following commands (feel free to use your own colors):
   - select @@bfactor >= 0.78
   - color sel medium violet red
   - select @@bfactor>=0.6 & @@bfactor < 0.78
   - color sel pale violet red
   - select @@bfactor < 0.6
   - color sel steel blue


[^1]: Local relax reference (https://elifesciences.org/articles/17219)
[^2]: DeepAccuracyNet reference (https://pubmed.ncbi.nlm.nih.gov/33637700/)
