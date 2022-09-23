""" a script that will use our model to predict errors for a map/pdb
    it will:
        run a local relax (optional)
        calculate density z-scores using my method
        run broken DAN to get lddts
        predict probabilities with the logistic regression model
        write the probabilties to a csv and a pdb file
"""
import pickle
import argparse
import os
import importlib.resources as pkg_resources
from math import floor

import medic.medic_model
import medic.broken_DAN as broken_DAN
import medic.density_zscores as dens_zscores
import medic.refine as refine
from medic.util import extract_energy_table
from medic.pdb_io import read_pdb_file, write_pdb_file


def compile_data(pdbf, mapf, reso, workers=100):
    print('\tcalculating zscores')
    data = dens_zscores.run(pdbf, mapf, reso)

    WINDOW_LENGTH = 20
    NEIGHBORHOOD = 20
    SCRIPT = "/home/reggiano/git/DeepAccNet/DeepAccNet.py"
    MEM = 40
    print('\tcalculating predicted lddts')
    #TODO - have this return a series, not a df??
    deepAccNet_scores = broken_DAN.calc_lddts(
                            pdbf,
                            WINDOW_LENGTH, WINDOW_LENGTH, NEIGHBORHOOD,
                            MEM, workers, SCRIPT)
    # NOTE - this naming convention comes from broken_DAN script
    data["lddt"] = deepAccNet_scores["lddt"]

    #TODO - return data series or something??
    # TODO - change this to pyrosetta thing instead??
    eng_dat = extract_energy_table(pdbf)
    data["rama_prepro"] = eng_dat["rama_prepro"]
    data["cart_bonded"] = eng_dat["cart_bonded"]
    return data


def get_feature_order():
    feats = list()
    params_str = pkg_resources.read_text(medic.medic_model, "model_params.txt")
    lines = [ x.strip() for x in params_str.split('\n') if x ]
    start = False
    for line in lines:
        if start:
            feats.append(line.split(':')[0].strip())
        if 'coefficients' in line:
            start = True
    return feats


def get_probabilities(df, model, prob_coln):
    order = get_feature_order()
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


def run_error_detection(pdbf, mapf, reso, run_relax=False):
    if run_relax:
        refined_pdb = f"{pdbf[:-4]}_refined.pdb"
        print("running local relax")
        refine.run(pdbf, mapf, reso, refined_pdb)
        pdbf = refined_pdb

    print("collecting scores")
    dat = compile_data(pdbf, mapf, reso, workers=20)

    print('loading statistical model')
    loaded_model = pickle.load(pkg_resources.open_binary(medic.medic_model, 'model.sav'))

    print('predicting errors')
    prob_coln = "error_probability"
    err_pred = get_probabilities(dat, loaded_model, prob_coln)
    err_pred.to_csv(f"{os.path.basename(pdbf)[:-4]}_MEDIC_predictions.csv")

    print('adding labels to bfactor in pdbs')
    set_pred_as_bfac(pdbf, err_pred, prob_coln)
    print('finished')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True)
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--reso', type=float, required=True)
    parser.add_argument('--relax', action="store_true", default=False, 
        help="run a relax before hand, pass model to the error detection")
    return parser.parse_args()


def commandline_main():
    args = parseargs()
    run_error_detection(args.pdb, args.map, args.reso, run_relax=args.relax)


if __name__ == "__main__":
    commandline_main()
