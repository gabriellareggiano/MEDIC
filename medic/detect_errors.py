""" a script that will try and use our model to predict errors for a map/pdb
    TODO - have it run a local relax?
    it will:
        calculate density z-scores using my method
        run broken DAN to get lddts
        predict probabilities with the provided logistic regression model
        make predictions based on provided threshold value
        write the errors to a csv and a pdb value 
        compare predicted errors to actual if provided?
"""
import pickle
import typing
import argparse
import os
import pandas as pd
import numpy as np
import subprocess
import shlex
from math import floor

import py_rosetta.validation_metrics.broken_DAN as broken_DAN
import py_rosetta.validation_metrics.run_denstools as dens_zscores
from py_rosetta.util_pyrosetta import extract_energy_table

from daimyo.core.pdb_io import read_pdb_file, write_pdb_file
from py_rosetta.rosetta_wrappers.relax import Relax

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True)
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--reso', type=float, required=True)
    parser.add_argument('--atom_mask', type=float, required=False, default=None)
    parser.add_argument('--feat_csv', type=str, required=False, 
        help='already collected the features? give the csv file')
    parser.add_argument('--ml_model', type=str, required=True,
        help="machine learned model from regression, saved in pickle/hdf5 file")
    parser.add_argument('--model_params', type=str, required=True,
        help='txt file that contains coefficients and intercept of linear model')
    parser.add_argument('--relax', action="store_true", default=False, 
        help="run a relax before hand, pass model to the error detection")
    return parser.parse_args()


def run_local_relax(pdbf, mapf, reso):
    relax = Relax(pdbf, mapf, reso, out_silent=False, 
                local_relax=True, bfactor=True, cart_min=True, 
                jobs=1, dens_wt=50.0, out_dir="")
    relax.setup_files(copy_input=False)
    cmd = relax.get_cmds()[0]
    expected_model = relax.get_model_fnames()[0]
    ret = subprocess.run(shlex.split(cmd),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
    if ret.returncode or not os.path.isfile(expected_model):
        raise RuntimeError(f"Failure with command: {cmd}\n")
    return expected_model


def compile_data(pdbf, mapf, reso, atom_mask, workers=100):
    print('calculating zscores')
    data = dens_zscores.run_denstools_perres(pdbf, mapf, reso,
                                     atom_mask=atom_mask)
    print('zscores calculated')
    WINDOW_LENGTH = 20
    NEIGHBORHOOD = 20
    SCRIPT = "/home/reggiano/git/DeepAccNet/DeepAccNet.py"
    MEM = 40
    print('running DAN')
    deepAccNet_scores = broken_DAN.calc_lddts(
                            pdbf,
                            WINDOW_LENGTH, WINDOW_LENGTH, NEIGHBORHOOD,
                            MEM, workers, SCRIPT)
    print('lddts calculated')
    # NOTE - this naming convention comes from broken_DAN script
    try: 
        data["lddt"] = deepAccNet_scores["lddt"]
    except KeyError:
        print('lddt not in dataframe, looking for mean_lddt') # currently never used
        data["lddt"] = deepAccNet_scores["mean_lddt"]
    # no longer optional
    eng_dat = extract_energy_table(pdbf)
    # HARD CODED geometry energies
    data["rama_prepro"] = eng_dat["rama_prepro"]
    data["cart_bonded"] = eng_dat["cart_bonded"]
    return data


#NOTEEEEEEEEEEEEEEEEE IF YOU CHANGE HOW WE WRITE THIS 
# IN TRAIN_MODEL YOU ARE SCREWED
def get_feature_order(txt_model_desc):
    feats = list()
    with open(txt_model_desc, 'r') as f:
        lines = [ x.strip() for x in f.readlines() ]
    start = False
    for line in lines:
        if start:
            feats.append(line.split(':')[0].strip())
        if 'coefficients' in line:
            start = True
    return feats


def calculate_normalized_probs(df, prob_col):
    resn_dat = {'resn': [],
                'cutoff': [],
                'dist_cutoff_ratio': []}
                
    # the following is hard coded from precision_recall_segments in regresh/2022_06_27/
    resn_dat['resn'] = ["ALA", "ARG", "ASN", "ASP", "CYS",
                        "GLU", "GLN", "GLY", "HIS", "ILE", 
                        "LEU", "LYS", "MET", "PHE", "PRO", 
                        "SER", "THR", "TRP", "TYR", "VAL"]
    resn_dat['cutoff'] = [0.811172, 0.807407, 0.796037, 0.824569, 0.787238, 
                          0.784476, 0.793658, 0.779986, 0.740723, 0.822984, 
                          0.762041, 0.780467, 0.824010, 0.765565, 0.887177, 
                          0.751814, 0.812239, 0.783968, 0.872570, 0.783127]
    resn_dat['dist_cutoff_ratio'] = [1.075720, 1.054691, 0.995896, 1.157867, 0.954712, 
                                     0.942475, 0.984416, 0.923240, 0.783431, 1.147503,
                                     0.853618, 0.925263, 1.154189, 0.866449, 1.800401,
                                     0.818443, 1.081832, 0.940259, 1.594026, 0.936613]
    resn_dat = pd.DataFrame.from_dict(resn_dat).set_index('resn')
    new_probs = list()
    for i in range(df.shape[0]):
        old_dist = 1.0 - df.at[i, prob_col]
        new_val = 1.0 - old_dist*resn_dat.at[df.at[i,'resn'], 'dist_cutoff_ratio']
        if new_val < 0: new_val = 0.0
        if new_val > 1: new_val = 1.0
        new_probs.append(new_val)
    df['normalized_prob'] = new_probs
    return df


def get_probabilities(df, model, model_desc):
    order = get_feature_order(model_desc)
    data = df[order]
    print('predicting probabilities')
    probabilities = model.predict_proba(data)
    df['prob'] = probabilities[:,1]
    print('calculating normalized probs')
    df = calculate_normalized_probs(df, 'prob')
    return df


# TODO - think about giving the option of true labels and using those somehow
def add_labels_to_pdb(pdbf, labels):
    # note - we end up editing this pose, i dont think it matters
    pose = read_pdb_file(pdbf)

    # write one with straight probabilities
    prob_cols = labels.filter(regex='prob')
    # match pdb_info
    labels['resID'] = labels["resi"].astype(str)+"."+labels["chID"]
    labels = labels.set_index("resID")
    for col in prob_cols:
        new_pdbf = f"{os.path.basename(pdbf)[:4]}_bf{col}.pdb"
        for ch in pose.chains:
            for grp in ch.groups:
                for atm in grp.atoms:
                    # want this to be rounded down (0.616 is not an error)
                    resid = f"{grp.groupNumber}.{ch.ID}"
                    try:
                        atm.bfactor = floor(labels.at[resid, col]*100)/100
                    except KeyError:
                        pass
        write_pdb_file(pose, new_pdbf)


def commandline_main():
    args = parseargs()
    if args.relax:
        print("running local relax")
        args.pdb = run_local_relax(args.pdb, args.map, args.reso)
    if not args.feat_csv:
        dat = compile_data(args.pdb, args.map, args.reso,
                           atom_mask=args.atom_mask,
                           workers=50)
    else:
        dat = pd.read_csv(args.feat_csv)
    print(f'loading model')
    loaded_model = pickle.load(open(args.ml_model,'rb'))
    err_lbls = get_probabilities(dat, loaded_model, args.model_params)
    print(f' errors predicted for all thresholds')
    err_lbls.to_csv(f"{os.path.basename(args.pdb)[:4]}_predictions.csv")
    print('adding labels to bfactor in pdbs')
    add_labels_to_pdb(args.pdb, err_lbls)
    print('finished')


if __name__ == "__main__":
    commandline_main()
