""" script to run DAN in smaller batches on a protein
TODO - turn this into a class and then run with apply?
"""

import numpy as np
import argparse
import os
import pandas as pd
import subprocess
from daimyo.core.pdb_io import read_pdb_file, write_pdb_file
from daimyo.core.util import get_number_of_residues
from more_itertools import consecutive_groups
import shlex
import pathlib
from time import sleep
from math import ceil
from copy import deepcopy

from dask_jobqueue import SLURMCluster
from dask.distributed import Client


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
    # don't think i need this lol
    # if len(new_resis) != len(resi_list):
    #    return close_gaps(new_resis, min_gap=10)
    return sorted(new_resis)


def remove_tiny(resi_list, min_len):
    """ some code to remove small pieces of residues,
        not currently in use
    """
    final_resis = sorted(resi_list.copy())
    for grp in consecutive_groups(final_resis):
        if len(list(grp)) < min_len:
            for g in grp:
                final_resis.remove(g)
    return final_resis


def same_chain_and_stragglers(residues, pinf, slide_len):
    """ make sure our main chain of residues is not split between chains
        and to also pick up any stragglers if they are just beyond the sliding dist
        i think the logic here is more complicated than it needs to be..........
    """
    ch_breaks = [i for i in range(1, len(pinf["chID"])) 
                    if pinf["chID"][i] != pinf["chID"][i-1]]
    for brk in ch_breaks:
        if residues[0] < pinf["ros_resi"][brk] < residues[-1]:
            if pinf["ros_resi"][brk] - residues[0] < residues[-1] - pinf["ros_resi"][brk]:
                residues = residues[residues.index(pinf["ros_resi"][brk]):]
            else:
                residues = residues[:residues.index(pinf["ros_resi"][brk])]
        elif pinf["ros_resi"][brk] < residues[0] and \
          residues[0] - pinf["ros_resi"][brk] <= ceil(slide_len/2):
            for ri in range(pinf["ros_resi"][brk], residues[0]):
                residues.append(ri)
        elif pinf["ros_resi"][brk] > residues[-1] and \
          pinf["ros_resi"][brk] - residues[-1] <= ceil(slide_len/2):
            for ri in range(residues[-1]+1, pinf["ros_resi"][brk]):
                residues.append(ri)
    return sorted(residues)


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


def run_DAN(script, pdb_file):
    """ run DeepAccNet on provided pdb
    """
    # NOTE - if you change name below, you will need to change line 197, 'index = ...'
    npz_file = pdb_file[:-4]+".npz"
    python_path = "/software/conda/envs/tensorflow/bin/python"
    cmd = f"{python_path} {script} --pdb {pdb_file} {npz_file}"
    ret = subprocess.run(shlex.split(cmd))
    if ret.returncode or not os.path.exists(npz_file):
        raise RuntimeError(f"DeepAccNet failed on {pdb_file}: {cmd}")
    return npz_file


# honestly this doesn't really work right now.. and idk if i care enough to 
# make this slightly faster quite yet. TODO here
def extract_and_run(pose, extract_residues, neighborhood, 
                    script_loc, pdbn, minimum_gap=10):
    extracted_pose, new_resis = extract_region(pose, 
                        extract_residues, 
                        neighborhood, min_gap=minimum_gap)
    ext_pdbf = f"{pdbn}_r{resi:04d}.pdb"
    extracted_pose.dump_pdb(ext_pdbf)
    log = f"{pdbn}_r{resi:04d}.log"
    npzf = run_DAN(script_loc, ext_pdbf, log)
    return npzf


def parseargs():
    parser = argparse.ArgumentParser()
    dan_params = parser.add_argument_group('DeepAccNet parameters')
    job_params = parser.add_argument_group('Slurm job parameters')
    dan_params.add_argument('--pdb', type=str)
    dan_params.add_argument('--window_length', type=int, default=20)
    dan_params.add_argument('--sliding_length', type=int, default=20)
    dan_params.add_argument('--neighborhood', type=float, default=20.0)
    dan_params.add_argument('--dan_script', type=str,
        default="/home/reggiano/git/DeepAccNet/DeepAccNet.py")
    job_params.add_argument('--queue', type=str, default='dimaio')
    job_params.add_argument('--memory', type=int, default=40)
    job_params.add_argument('--num_workers', type=int, default=100)
    return parser.parse_args()


def calc_lddts(pdbf, win_len, slide_len, neighborhood, 
               mem, workers, script_loc):
    full_pose = read_pdb_file(pdbf)
    pinf = {"resn": list(),
            "resi": list(),
            "chID": list(),
            "ros_resi": list() }
    total_residues = get_number_of_residues(full_pose)
    resi = 1
    for ch in full_pose.chains:
        for grp in ch.groups:
            pinf["resn"].append(grp.groupName)
            pinf["resi"].append(grp.groupNumber)
            pinf["chID"].append(ch.ID)
            pinf["ros_resi"].append(resi)
            resi += 1
    full_results = pd.DataFrame.from_dict(pinf)
    full_results['lddt'] = np.nan

    print('submitting tasks for DeepAccNet')
    tasks = list()
    all_main_residues = list()
    all_extracted_residues = list()
    task_identifier = list()
    with SLURMCluster(
        cores=1,
        memory=f"{mem}GB",
        queue="cpu",
        walltime="01:00:00", # this should in theory by plenty of time
        job_name=f"{os.path.basename(pdbf)[:4]}_DAN",
        env_extra=["source ~/.bashrc"]
    ) as cluster:
        cluster.adapt(minimum=0, maximum=workers)
        with Client(cluster) as client:
            for resi in range(1, total_residues, slide_len):
                task_identifier.append(resi)
                end = resi+win_len
                if end > total_residues:
                    end = total_residues+1
                main_residues = list(range(resi,end))
                main_residues = same_chain_and_stragglers(main_residues, pinf, slide_len)
                all_main_residues.append(main_residues)
                extracted_pose, new_resis = extract_region(full_pose,
                                    main_residues, pinf,
                                    neighborhood)
                all_extracted_residues.append(new_resis)
                # NOTE - if you change name below you need to change line 197, 'index = ...'
                ext_pdbf = f"{os.path.basename(pdbf)[:-4]}_r{resi:04d}.pdb" 
                write_pdb_file(extracted_pose, ext_pdbf)
                tasks.append(client.submit(run_DAN, script_loc, ext_pdbf))

            # i think this might mean that all the lists 
            # i made for the next part are unnecessary or at least the indexing part is
            print('gathering results')
            
            sleep(60) # before running any DAN tasks, wait a bit before reading pdbs
            results = client.gather(tasks)

    sleep(30) # after closing client, wait a bit before trying to read files
    print('collecting data')
    for result in results:
        npz_data = np.load(result)
        index = task_identifier.index(int(result.split('_')[-1][:-4].strip('r')))
        curr_col = f"lddt_run{index:03d}"
        df_data = pd.DataFrame.from_dict(npz_data["lddt"])
        df_data["ros_resi"] = all_extracted_residues[index]
        df_data.rename({0:'lddt'}, axis=1, inplace=True)
        main_lddts = df_data[df_data['ros_resi'].isin(all_main_residues[index])]
        if slide_len != win_len: # some overlap
            full_results[curr_col] = full_results['ros_resi'].map(main_lddts.set_index('ros_resi')['lddt'])
        else:
            # i think this is dumb way to do this but i'm too lazy right now to find the better way
            full_results['lddt'] = full_results['ros_resi'].map(main_lddts.set_index('ros_resi')['lddt']).fillna(full_results['lddt'])
    
    print('all data collected')
    if slide_len != win_len:
        full_results["mean_lddt"] = full_results.filter(regex='lddt_run').mean(skipna=True, axis=1)
    return full_results


def commandline_main():
    args = parseargs()
    df = calc_lddts(args.pdb, args.window_length, 
        args.sliding_length, args.neighborhood, 
        args.memory, args.num_workers,
        args.dan_script)
    df.to_csv(f"{os.path.basename(args.pdb)[:4]}_DAN.csv")


if __name__ == "__main__":
    commandline_main()
