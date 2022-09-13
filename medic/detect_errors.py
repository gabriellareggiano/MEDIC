""" a script that will use our model to predict errors for a map/pdb
    it will:
        run a local relax (optional)
        calculate density z-scores using my method
        run broken DAN to get lddts
        predict probabilities with the provided logistic regression model
        write the probabilties to a csv and a pdb file
"""
import pyrosetta
import pickle
import typing
import argparse
import os
import importlib.resources as pkg_resources
from math import floor

import medic.broken_DAN as broken_DAN
import medic.run_denstools as dens_zscores
from medic.util import extract_energy_table
from medic.pdb_io import read_pdb_file, write_pdb_file
from medic.relax import Relax


def run_local_relax(pdbf, relax):
    out_pdb = f"{pdbf[:-4]}_0001.pdb"
    pose = pyrosetta.pose_from_file(pdbf)
    relax.apply(pose)
    pose.dump_pdb(out_pdb)
    return out_pdb


def compile_data(pdbf, mapf, reso, workers=100):
    print('calculating zscores')
    data = dens_zscores.run_denstools_perres(pdbf, mapf, reso)
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
    eng_dat = extract_energy_table(pdbf)
    data["rama_prepro"] = eng_dat["rama_prepro"]
    data["cart_bonded"] = eng_dat["cart_bonded"]
    return data


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


def get_probabilities(df, model, model_desc, prob_coln):
    order = get_feature_order(model_desc)
    data = df[order]
    print('predicting probabilities')
    probabilities = model.predict_proba(data)
    df[prob_coln] = probabilities[:,1]
    return df


def set_pred_as_bfac(pdbf, predictions, prob_coln):
    pose = read_pdb_file(pdbf)
    predictions['resID'] = predictions["resi"].astype(str)+"."+predictions["chID"]
    predictions = predictions.set_index("resID")
    new_pdbf = f"{os.path.basename(pdbf)[:4]}_MEDIC_bfac_pred.pdb"
    for ch in pose.chains:
        for grp in ch.groups:
            for atm in grp.atoms:
                resid = f"{grp.groupNumber}.{ch.ID}"
                try:
                    #TODO - do actual rounding? instead of auto flooring?
                    atm.bfactor = floor(predictions.at[resid, prob_coln]*100)/100
                except KeyError:
                    pass
    write_pdb_file(pose, new_pdbf)


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True)
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--reso', type=float, required=True)
    parser.add_argument('--ml_model', type=str, required=True,
        help="machine learned model from regression, saved in pickle/hdf5 file")
    parser.add_argument('--model_params', type=str, required=True,
        help='txt file that contains coefficients and intercept of linear model')
    parser.add_argument('--relax', action="store_true", default=False, 
        help="run a relax before hand, pass model to the error detection")
    return parser.parse_args()


def commandline_main():
    args = parseargs()
    if args.relax:
        print("running local relax")
        relax = Relax(args.map, args.reso)
        pyrosetta.init(relax.get_flag_file_str())
        args.pdb = run_local_relax(args.pdb, relax)
    dat = compile_data(args.pdb, args.map, args.reso,
                           workers=50)
    print(f'loading model')
    loaded_model = pickle.load(open(args.ml_model,'rb'))
    prob_coln = "error_probability"
    err_pred = get_probabilities(dat, loaded_model, args.model_params, prob_coln)
    print(f' errors predicted for all thresholds')
    err_pred.to_csv(f"{os.path.basename(args.pdb)[:-4]}_MEDIC_predictions.csv")
    print('adding labels to bfactor in pdbs')
    set_pred_as_bfac(args.pdb, err_pred, prob_coln)
    print('finished')


if __name__ == "__main__":
    commandline_main()
