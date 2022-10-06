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
import medic.analysis as analyze
from medic.util import extract_energy_table, clean_dan_files, clean_pdb
from medic.pdb_io import read_pdb_file, write_pdb_file

def compile_data(pdbf, mapf, reso, verbose=False,
        mem=0, queue="", workers=0):
    if verbose: print('calculating zscores')
    data = dens_zscores.run(pdbf, mapf, reso)

    WINDOW_LENGTH = 20
    NEIGHBORHOOD = 20
    if verbose: print('calculating predicted lddts')
    if mem and queue and workers:
        deepAccNet_scores = broken_DAN.calc_lddts_hpc(pdbf, WINDOW_LENGTH,
                                NEIGHBORHOOD,
                                mem, queue, workers,
                                verbose=verbose)
    else: # run locally
        deepAccNet_scores = broken_DAN.calc_lddts(pdbf, WINDOW_LENGTH, 
                                NEIGHBORHOOD, verbose=verbose)
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


def run_error_detection(pdbf, mapf, reso, verbose=False,
                mem=0, queue="", workers=0):
    if verbose: print("collecting scores")
    dat = compile_data(pdbf, mapf, reso, verbose=verbose,
                    mem=mem, queue=queue, workers=workers)

    if verbose: print('loading statistical model')
    loaded_model = pickle.load(pkg_resources.open_binary(medic.medic_model, 'model.sav'))

    if verbose: print('predicting errors')
    prob_coln = "error_probability"
    err_pred = get_probabilities(dat, loaded_model, prob_coln)

    return err_pred


def parseargs():
    parser = argparse.ArgumentParser()
    inp_params = parser.add_argument_group('Inputs')
    options = parser.add_argument_group("Options")
    job_params = parser.add_argument_group('Running DeepAccNet on HPC')
    inp_params.add_argument('--pdb', type=str, required=True)
    inp_params.add_argument('--map', type=str, required=True)
    inp_params.add_argument('--reso', type=float, required=True)
    options.add_argument('--clean', action="store_true", default=False,
        help="clean the pdb before running (remove HETATMS/noncanonicals)")
    options.add_argument('--skip_relax', action="store_true", default=False, 
        help="skip the relax step, only do this if your pdb is from Rosetta")
    options.add_argument('--keep_intermediates', action="store_true", default=False,
        help="dont clean up temperorary files created for deepaccnet")
    options.add_argument('-v','--verbose', action="store_true", default=False,
        help="print extra updates")
#    options.add_argument('-j', '--processors', type=int, required=False, default=1,
#        help='number processors to use if running locally')
    job_params.add_argument('--scheduler', action="store_true", default=False,
        help='submit to cluster, uses slurm scheduler through dask')
    job_params.add_argument('--queue', type=str, default="", required=False,
        help="queue to run on")
    job_params.add_argument('--workers', type=int, default=0, required=False,
        help="number of workers to run DAN on concurrently")
    return parser.parse_args()


def commandline_main():
    args = parseargs()

    if args.clean:
        args.pdb = clean_pdb(args.pdb)

    if not args.skip_relax:
        refined_pdb = f"{args.pdb[:-4]}_refined.pdb"
        if args.verbose: print("running local relax")
        refine.run(args.pdb, args.map, args.reso, refined_pdb)
        args.pdb = refined_pdb

    MEM = 0
    if args.scheduler:
        MEM = 40
        if not args.queue:
            raise RuntimeError('set queue to run with scheduler')
        if not args.workers:
            raise RuntimeError('specify number of workers to use with dask')
        errors = run_error_detection(args.pdb, args.map, args.reso,
                        mem=MEM, queue=args.queue, workers=args.workers, verbose=args.verbose)
    else:
        errors = run_error_detection(args.pdb, args.map, args.reso, verbose=args.verbose)
    
    prob_coln = "error_probability" # this is defined in two places ugly

    # analysis
    high_error_threshold = 0.78
    low_error_threshold = 0.60
    if args.verbose: print('calculating score contributions')
    error_dict_summary = analyze.collect_error_info(errors, prob_coln, low_error_threshold, 
                            contributing_csvf=f"{os.path.basename(args.pdb)[:-4]}_MEDIC_predictions.csv")
    error_summary = analyze.get_error_report_str(error_dict_summary, 
                        high_error_threshold, 
                        low_error_threshold)
    
    # output files
    set_pred_as_bfac(args.pdb, errors, prob_coln)
    with open("MEDIC_summary.txt", 'w') as f:
        f.write(error_summary)

    # print analysis
    print('\n\n')
    print('------------------------- ERROR SUMMARY -------------------------\n')
    print(error_summary)
    print('\n-----------------------------------------------------------------')
    
    if not args.keep_intermediates:
        clean_dan_files(args.pdb)

    
if __name__ == "__main__":
    commandline_main()
