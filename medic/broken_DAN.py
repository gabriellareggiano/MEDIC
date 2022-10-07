""" script to run DAN in smaller batches on a protein
"""

import numpy as np
import argparse
import os
import pandas as pd
from more_itertools import consecutive_groups
from time import sleep
from math import ceil
from copy import deepcopy
import torch
import sys
import multiprocessing

from medic.pdb_io import read_pdb_file, write_pdb_file
from medic.util import get_number_of_residues
import DeepAccNet.deepAccNet as dan
import pyrosetta

def close_gaps(resi_list, min_gap):
    """ after extracting nearby residues, want to make sure
        that there are no small gaps between stretches
    """
    new_resis = sorted(resi_list.copy())
    consec_resis = [list(item) for item in consecutive_groups(new_resis)]
    for i, grp1 in enumerate(consec_resis[:-1]):
        start = grp1[-1]
        end = consec_resis[i+1][0]
        if abs(start - end) <= min_gap:
            for add_res in range(start+1,end):
                new_resis.append(add_res)
    return sorted(new_resis)


def same_chain_and_stragglers(residues, pinf, slide_len, seen):
    """ make sure our main chain of residues is not split between chains
        and to also pick up any stragglers if they are just beyond the sliding dist
        i think the logic here is more complicated than it needs to be..........
    """
    new_residues = [i for i in residues]
    ch_breaks = [i for i in range(1, len(pinf["chID"])) 
                    if pinf["chID"][i] != pinf["chID"][i-1]]
    for brk in ch_breaks:
        if residues[0] < pinf["ros_resi"][brk] < residues[-1]:
            if pinf["ros_resi"][brk] - residues[0] < residues[-1] - pinf["ros_resi"][brk]:
                new_residues = residues[residues.index(pinf["ros_resi"][brk]):]
            else:
                new_residues = residues[:residues.index(pinf["ros_resi"][brk])]
        elif pinf["ros_resi"][brk] < residues[0] and \
          residues[0] - pinf["ros_resi"][brk] <= ceil(slide_len/2):
            for ri in range(pinf["ros_resi"][brk], residues[0]):
                if seen[ri] == 0:
                    new_residues.append(ri)
        elif pinf["ros_resi"][brk] > residues[-1] and \
          pinf["ros_resi"][brk] - residues[-1] <= ceil(slide_len/2):
            for ri in range(residues[-1]+1, pinf["ros_resi"][brk]):
                if seen[ri] == 0:
                    new_residues.append(ri)
    for ri in new_residues:
        seen[ri] = 1
    return sorted(new_residues)


def find_context(pose, ca_atm, nbrhd):
    """ find residues within the designated distance
        uses rosetta numbering right now because of stupid reasons
    """
    context_resi = list()
    resi = 1
    for ch in pose.chains:
        for grp in ch.groups:
            if ca_atm.distance(grp.atom_by_name("CA")) <= nbrhd:
                context_resi.append(resi)
            resi += 1
    return context_resi


def extract_region(pose, extract_resis, pdbinfo, 
                nbrhood, min_gap = 10, min_len=5):
    """ given a pose and the provided residues,
        find all the residues around the provided ones,
        delete all the regions we dont care about
        return new pose
    """
    new_pose = deepcopy(pose)
    context_resis = list()
    for resi in extract_resis:
        pinf_ind = pdbinfo["ros_resi"].index(resi)
        chain_id = pdbinfo["chID"][pinf_ind]
        res_id = pdbinfo["resi"][pinf_ind]
        for ch in [x for x in pose.chains if x.ID == chain_id]:
            for grp in [y for y in ch.groups if y.groupNumber == res_id]:
                alphac = grp.atom_by_name("CA")
        context_resis += find_context(pose, alphac, nbrhood)
    keep_resis = close_gaps(sorted(list(set(extract_resis + context_resis))), min_gap)
    to_delete = list()
    resi = 1
    for i, ch in enumerate(pose.chains):
        for j, grp in enumerate(ch.groups):
            if resi not in keep_resis:
                to_delete.append((i, j))
            resi += 1
    for (i, j) in reversed(to_delete):
        del new_pose.chains[i].groups[j]
    return new_pose, keep_resis


def run_dan(infilepath):
    script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DeepAccNet")
    modelpath = os.path.join(script_dir, "models", "NatComm_standard")

    outfilepath = f"{infilepath[:-4]}.npz"
    infolder = "/".join(infilepath.split("/")[:-1])
    insamplename = infilepath.split("/")[-1][:-4]
    outfolder = "/".join(outfilepath.split("/")[:-1])
    outsamplename = outfilepath.split("/")[-1][:-4]
    feature_file_name = os.path.join(outfolder, outsamplename+".features.npz")

    if (not os.path.isfile(feature_file_name)):
        dan.process((os.path.join(infolder, insamplename+".pdb"),
                            feature_file_name, False))
        
    if os.path.isfile(feature_file_name):
        # Load pytorch model:
        #model = dan.DeepAccNet()
        model = dan.DeepAccNet(twobody_size = 33)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model.load_state_dict(torch.load("models/regular_rep1/weights.pkl"))
        model.load_state_dict(torch.load(os.path.join(modelpath, "best.pkl"), map_location=device)['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Actual prediction
        with torch.no_grad():
            (idx, val), (f1d, bert), f2d, dmy = dan.getData(feature_file_name)
            f1d = torch.Tensor(f1d).to(device)
            f2d = torch.Tensor(np.expand_dims(f2d.transpose(2,0,1), 0)).to(device)
            idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
            val = torch.Tensor(val).to(device)

            estogram, mask, lddt, dmy = model(idx, val, f1d, f2d)
            lddt_cpu = lddt.cpu().detach().numpy().astype(np.float16)
            np.savez_compressed(outsamplename+".npz",
                    lddt = lddt_cpu,
                    estogram = estogram.cpu().detach().numpy().astype(np.float16),
                    mask = mask.cpu().detach().numpy().astype(np.float16))

        dan.clean([outsamplename],
                    outfolder,
                    verbose=False,
                    ensemble=False)
    else:
        print(f"Feature file does not exist: {feature_file_name}", file=sys.stderr)

    return lddt_cpu


def calc_lddts(pdbf, win_len, neighborhood, verbose=False, processes=1):
    pyrosetta.init("-ex1 -ex2aro -constant_seed -read_only_ATOM_entries")
    pyrosetta.rosetta.basic.options.set_boolean_option("in:missing_density_to_jump", False)
    pyrosetta.rosetta.basic.options.set_boolean_option("cryst:crystal_refine", False)
    pyrosetta.rosetta.basic.options.set_boolean_option("corrections:shapovalov_lib_fixes_enable", False)
    pyrosetta.rosetta.basic.options.set_file_option("corrections:score:rama_pp_map","scoring/score_functions/rama/fd")
    full_pose = read_pdb_file(pdbf)
    pinf = {"resn": list(),
            "resi": list(),
            "chID": list(),
            "ros_resi": list() }
    total_residues = get_number_of_residues(full_pose)
    resi = 1
    seen_residues = dict()
    for ch in full_pose.chains:
        for grp in ch.groups:
            pinf["resn"].append(grp.groupName)
            pinf["resi"].append(grp.groupNumber)
            pinf["chID"].append(ch.ID)
            pinf["ros_resi"].append(resi)
            seen_residues[resi] = 0
            resi += 1
    full_results = pd.DataFrame.from_dict(pinf)
    full_results['lddt'] = np.nan

    if verbose: print(f'running tasks for DeepAccNet on {processes} processes')

    # setup for dan
    indices_to_keep = list()
    extracted_lddts = list()
    inputs = list()
    for resi in range(1, total_residues, win_len):
        end = resi+win_len
        if end >= total_residues:
            end = total_residues+1
        init_residues = list(range(resi,end))
        main_residues = same_chain_and_stragglers(init_residues, pinf, win_len, seen_residues)
        extracted_pose, new_resis = extract_region(full_pose,
                            main_residues, pinf,
                            neighborhood)
        indices_to_keep.append((new_resis.index(main_residues[0]),
                            new_resis.index(main_residues[-1])))
        ext_pdbf = f"{os.path.basename(pdbf)[:-4]}_r{resi:04d}.pdb" 
        write_pdb_file(extracted_pose, ext_pdbf)
        inputs.append(ext_pdbf)
    
    # run dans
    if processes > 1:
        with multiprocessing.Pool(processes) as pool:
            extracted_lddts = pool.map(run_dan, inputs)
    else:
        for p in inputs:
            extracted_lddts.append(run_dan(p))
    
    main_lddts = list()
    for i,lddts in enumerate(extracted_lddts):
        main_lddts.append(lddts[indices_to_keep[i][0]:indices_to_keep[i][1]+1])
    main_lddts = np.concatenate(main_lddts)

    full_results['lddt'] = main_lddts
    if verbose: print('all data collected')
    return full_results


def calc_lddts_hpc(pdbf, win_len,neighborhood, 
                mem, queue, workers, verbose=False):
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    pyrosetta.init("-ex1 -ex2aro -constant_seed -read_only_ATOM_entries -mute all")
    pyrosetta.rosetta.basic.options.set_boolean_option("corrections:beta", False)
    pyrosetta.rosetta.basic.options.set_boolean_option("corrections:beta_cart", False)
    full_pose = read_pdb_file(pdbf)
    pinf = {"resn": list(),
            "resi": list(),
            "chID": list(),
            "ros_resi": list() }
    total_residues = get_number_of_residues(full_pose)
    resi = 1
    seen_residues = dict()
    for ch in full_pose.chains:
        for grp in ch.groups:
            pinf["resn"].append(grp.groupName)
            pinf["resi"].append(grp.groupNumber)
            pinf["chID"].append(ch.ID)
            pinf["ros_resi"].append(resi)
            seen_residues[resi] = 0
            resi += 1
    full_results = pd.DataFrame.from_dict(pinf)
    full_results['lddt'] = np.nan

    if verbose: print('submitting tasks for DeepAccNet')
    tasks = list()
    indices_to_keep = list()
    with SLURMCluster(
        cores=1,
        memory=f"{mem}GB",
        queue=queue,
        walltime="01:00:00", # this should in theory by plenty of time
        job_name=f"{os.path.basename(pdbf)[:4]}_DAN"
    ) as cluster:
        cluster.adapt(minimum=0, maximum=workers)
        with Client(cluster) as client:
            for resi in range(1, total_residues, win_len):
                end = resi+win_len
                if end >= total_residues:
                    end = total_residues+1
                init_residues = list(range(resi,end))
                main_residues = same_chain_and_stragglers(init_residues, pinf, win_len, seen_residues)
                extracted_pose, new_resis = extract_region(full_pose,
                                    main_residues, pinf,
                                    neighborhood)
                indices_to_keep.append((new_resis.index(main_residues[0]),
                                   new_resis.index(main_residues[-1])))
                # NOTE - if you change name below you need to change line 197, 'index = ...'
                ext_pdbf = f"{os.path.basename(pdbf)[:-4]}_r{resi:04d}.pdb" 
                write_pdb_file(extracted_pose, ext_pdbf)
                tasks.append(client.submit(run_dan, ext_pdbf))

            if verbose: print('gathering results')
            
            sleep(30) # before running any DAN tasks, wait a bit before reading pdbs
            extracted_lddts = client.gather(tasks)

    sleep(30) # after closing client, wait a bit before trying to read files
    if verbose: print('collecting data')
    main_lddts = list()
    for i,lddts in enumerate(extracted_lddts):
        main_lddts.append(lddts[indices_to_keep[i][0]:indices_to_keep[i][1]+1])
    main_lddts = np.concatenate(main_lddts)
    
    full_results['lddt'] = main_lddts
    if verbose: print('all data collected')
    return full_results


def parseargs():
    parser = argparse.ArgumentParser()
    dan_params = parser.add_argument_group('DeepAccNet parameters')
    job_params = parser.add_argument_group('Slurm job parameters')
    dan_params.add_argument('--pdb', type=str)
    dan_params.add_argument('--window_length', type=int, default=20)
    dan_params.add_argument('--sliding_length', type=int, default=20)
    dan_params.add_argument('--neighborhood', type=float, default=20.0)
    job_params.add_argument('--queue', type=str, default='dimaio')
    job_params.add_argument('--memory', type=int, default=40)
    job_params.add_argument('--num_workers', type=int, default=100)
    return parser.parse_args()


def commandline_main():
    args = parseargs()
    df = calc_lddts_hpc(args.pdb, args.window_length, args.neighborhood, 
        args.memory, args.queue, args.num_workers)
    df.to_csv(f"{os.path.basename(args.pdb)[:4]}_DAN.csv")


if __name__ == "__main__":
    commandline_main()

