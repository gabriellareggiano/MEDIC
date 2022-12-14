# MEDIC: Model error detection in cryo-EM

MEDIC is a statistical model derived from logistic regression that will identify possible errors in your structure. It will predict a probability of error (high value = more likely to be an error) for every residue.

## Manuscripts

Robust residue-level error detection in cryo-electron microscopy models. Gabriella Reggiano, Daniel Farrell, Frank DiMaio (https://www.biorxiv.org/content/10.1101/2022.09.12.507680v1)

## Data availability

You can download the structures used for training and validation [here](https://files.ipd.uw.edu/pub/MEDIC/errors.tar.gz)

You can find errors identified by MEDIC on deposited EM structures between 3-5A resolution [here](https://files.ipd.uw.edu/pub/MEDIC/emdb_curated_MEDIC_errors.csv)

## Computational Resources
DeepAccuracyNet runs much faster on GPUs, so if you have one available, we recommend using one to run MEDIC. We have run it on GPUs with only 8GB GPU memory.

MEDIC has multiprocessing built in, so multiple cores can be used to speed up predictions.

We have tested MEDIC on a 2800 residue structure and were able to run it on a personal laptop. 

## Installation/dependencies

To install MEDIC, you need to:
- install anaconda and pip on your system
- get a license for pyrosetta
- create a python environment for MEDIC
- git clone the source code and install MEDIC

#### PyRosetta License
In depth instructions for the installation of pyrosetta can be found here: [Installation with environment manager](https://www.pyrosetta.org/downloads#h.6vttn15ac69d).
Or you can follow the instructions below.

- Apply for a license (free for academic use) [here](https://els2.comotion.uw.edu/product/rosetta)
- Add the PyRosetta channel to your *~/.condarc* and replace the *username* and *password* with your information
```
channels:
    - https://username:password@conda.graylab.jhu.edu
    - defaults
```

#### Create a conda environment for MEDIC (here called *medic*):
```
conda create -n medic -y pyrosetta
```

if you are using a Mac:
```
conda create -n medic -y python=3.9 pyrosetta pytorch=1.10
```

#### Install MEDIC into the active conda environment
```
conda activate medic
git clone --recursive https://github.com/gabriellareggiano/MEDIC.git
cd MEDIC
pip install -e .
```


## Running MEDIC on your structure
### MEDIC background

MEDIC has four steps, all performed with this script:
1. Local relax in Rosetta [^1]
2. Calculation of density z-scores
3. Calculation of predicted lDDTs [^2]
4. Error prediction

### Running MEDIC

You can run the following to see all options for MEDIC
```
./path/to/MEDIC/detect_errors.py --help
```
\
MEDIC should be installed as an executable script in your bin. If you run:
```
which detect_errors.py
```
and get a path to a file, then you can run MEDIC with only:
```
./detect_errors.py --help
```
\
The minimal command is shown below. Make sure your pdb is docked into the map before running.
```
./path/to/MEDIC/detect_errors.py --pdb {path/to/pdb} --map {path/to/map} -???reso {global resolution} ???j {number_processes}
```
Increasing the number of processes with `-j` will make step 3 go faster. However, if you run out of memory, lower the number of processes.

If your structure has already been relaxed with Rosetta, add the flag: `--skip_relax`
  - The relax is mandatory, don???t skip if your pdb hasn???t been through Rosetta

If your structure has ligands or nucleac acids or noncanonical amino acids, add the flag: `--clean` 

## Visualizing and inspecting outputs

### MEDIC outputs the following files:

- `{pdb}_refined.pdb`
  - this is your structure after the local relax

- `{pdb}_MEDIC_bfac_pred.pdb`
  - this is your relaxed structure with the probabilities in the B-factor column

- `{pdb}_MEDIC_predictions.csv`
  - this contains all the relevant scores for every residue with the predicted probabilities

- `MEDIC_summary_{pdb}.txt`
  - this contains all the segments that have been marked as errors, as well as the scores that flagged them as errors
```
              25R - 28R, definite error       ???> high probability error, residues 25-28, chain R
                   causes: density               ???> density score alone can predict this to be an error

              23R - 24R, possible error     ???> low probability error, residues 23-24, chain R
                   causes: density + lddt     ???> the density and the lddt together predict an error
```

### To view your error predictions in Chimera:

1. Load in your `*bfac_pred.pdb` to Chimera. 
2. Go to Tools -> Depiction -> Render by attribute
3. In the Render by attribute window:
   - Attributes of -> residues
   - Attribute -> average -> bfactor
   - Choose your thresholds for coloring we recommend:
     - 0.78 -> far right histogram
     - 0.60 -> left
   - Note that setting this value to lower than 0.60 may allow you to find more errors, but MEDIC will also mark more false positives.


### To view your error predictions in ChimeraX:

1. Load in your `*bfac_pred.pdb` to Chimera
2. Enter the following commands (feel free to use your own colors):
   - `select @@bfactor >= 0.78`
   - `color sel medium violet red`
   - `select @@bfactor>=0.6 & @@bfactor < 0.78`
   - `color sel pale violet red`
   - `select @@bfactor < 0.6`
   - `color sel steel blue`


[^1]: Local relax reference (https://elifesciences.org/articles/17219)
[^2]: DeepAccuracyNet reference (https://pubmed.ncbi.nlm.nih.gov/33637700/)
