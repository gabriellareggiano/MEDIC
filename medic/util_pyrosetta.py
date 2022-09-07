#!/usr/bin/env python
"""home for useful functions for pyrosetta scripts
"""
from daimyo.core.pdb_io import read_pdb_lines, write_pdb_file
import typing
import numpy as np
import pandas as pd
import daimyo.core.pose
import math
from daimyo.core.util import get_CA_coords, get_number_of_residues
from operator import itemgetter
import glob
import os
import pathlib

def dump_coords(all_xyz, pdb_name):
    """ dump list of xyzVectors to pdb
    """
    import pyrosetta
    tmp_pose = pyrosetta.rosetta.core.pose.Pose()
    rsd_set = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance().residue_type_set("fa_standard")
    for i in all_xyz:
        mgres = pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue( rsd_set.get_representative_type_name3(" MG"))
        mgres.set_xyz("MG",i)
        tmp_pose.append_residue_by_jump(mgres,1)
    if pdb_name[-4:] != ".pdb":
        pdb_name += ".pdb" 
    tmp_pose.dump_pdb(pdb_name)


def extract_energy_table(pdbf):
    """ put the energy table from rosetta at the end of a pdb
        into a dataframe
    """
    with open(pdbf, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    header = 0
    end_line = -2
    for i,line in enumerate(reversed(lines)):
        if "label" == line[0:5]:
            header = i
            break
        if "#END_POSE" in line:
            end_line = i
    header = len(lines) - header - 1
    end_line = len(lines) - header - end_line - 2
    energy_table = pd.read_csv(pdbf, header=header, sep="\s+").iloc[2:end_line]
    energy_table = energy_table[~energy_table['label'].str.contains('VRT')]
    energy_table = energy_table.reset_index()
    return energy_table


# TODO - this basically in iteration analysis should probably fix that
def pick_topn_pdbs(contain_directory, out_directory = "top_pdbs",
                   pdb_basename = "", score_type = "",
                   n=5):
    """ pick top pdbs when there are no sc files
        and you need to parse actual pdb files
        give pdb name if multiple groups in your directory
    """
    pdbs = glob.glob(os.path.join(contain_directory,f"{pdb_basename}*pdb"))
    if len(pdbs) == 0:
        raise RuntimeError(f"no pdbs with base name '{pdb_basename}' in {contain_directory}")
    scores = list()
    for i, pdb in enumerate(pdbs):
        with open(pdb, "r") as f:
            lines = [line.strip() for line in f.read().split('\n')]
        col_num = -1
        score_line = ""
        for line in reversed(lines):
            if not line: continue
            if line.split()[0] == "label":
                if score_type:
                    for j, sc_type in enumerate(line.split()):
                        if sc_type.strip() == score_type:
                            col_num = j
                break   # we care about nothing after label line
            if line.split()[0] == "pose":
                score_line = line.split()
        if not score_line: continue
        scores.append((i,float(score_line[col_num].strip())))
    sorted_scores = sorted(scores, key=lambda x: x[1])
    pathlib.Path(out_directory).mkdir(exist_ok=True, parents=True)
    final_pdbs = list()
    for i in range(n):
        pdb_name = f"{pdb_basename}_rank_{i+1:02d}.pdb" if pdb_basename else f"rank_{i+1:02d}.pdb"
        sl_name = os.path.join(out_directory, pdb_name)
        # if it starts with mnt it'll mess shit up
        if os.path.abspath(pdbs[sorted_scores[i][0]])[:4] == "/mnt":
            pdb_path = os.path.abspath(pdbs[sorted_scores[i][0]])[4:]
        else:
            pdb_path = os.path.abspath(pdbs[sorted_scores[i][0]])
        os.symlink(pdb_path, sl_name)
        final_pdbs.append(pdbs[sorted_scores[i][0]])
    return final_pdbs


def renumber_pose(pose: daimyo.core.pose.Pose) -> daimyo.core.pose.Pose:
    """ renumber the residues so that we start from 1 
    """
    resnum = 1
    for chain in pose.chains:
        for group in chain.groups:
            group.groupNumber = resnum
            resnum += 1
    return pose


def remove_res_range(pdb: str, range_list: typing.List[int], out_pdb: str = "") -> None:
    """ this is basically a carbon copy of dan's script
        but that is only written as a main function with argparse so
        here we are
            param range_list: list of start, end, start, end, start, end for deletion
    """
    if not out_pdb:
        out_pdb = f"{pdb[:-4]}_trimmed.pdb"
    with open(pdb, 'r') as pdbf:
        pose = read_pdb_lines(pdbf.read().split('\n'))
    pose = renumber_pose(pose)
    to_delete = []
    begin_ends = list(zip(*[iter(range_list)]*2))
    for i, chain in enumerate(pose.chains):
        for j, group in enumerate(chain.groups):
            if any(begin < group.groupNumber < end for begin, end in begin_ends):
                to_delete.append((i, j))
    for (i, j) in reversed(to_delete):
        del pose.chains[i].groups[j]

    write_pdb_file(pose, out_pdb)


def rename_chains(pdb: str, out_pdb: str = "") -> None:
    """ rename chains so that each chain has its own letter
        lazily assumes you won't have more than 26 chains
    """
    if not out_pdb:
        out_pdb = f"{pdb[:-4]}_chs.pdb"
    with open(pdb, 'r') as pdbf:
        pose = read_pdb_lines(pdbf.read().split('\n'))
    for i, ch in enumerate(pose.chains):
        new_chain = chr(ord('@')+i+1)
        ch.name = new_chain
        ch.ID = new_chain
    write_pdb_file(pose, out_pdb)


def rmsd_CA_noalign(CAs1: typing.Dict[int,np.ndarray],
                    CAs2: typing.Dict[int,np.ndarray]) -> float:
    """ calculate CA rmsd for two dicts of CA coords (output from dan's code)
        without aligning them. 
    """
    r1_coords = np.array([v for v in CAs1.values()])
    r2_coords = np.array([v for v in CAs2.values()])
    diff = r1_coords - r2_coords
    N = r1_coords.shape[0]
    return np.sqrt((diff * diff).sum() / N)


def get_specific_CAs(pose: daimyo.core.pose.Pose,
                    range_list:typing.List[int]) -> typing.Dict[int,np.ndarray]:
    """ get only CAs in a specific range
        range list should be length of 2
        it is inclusive for range ends
    """
    CA_coords = dict()
    chain_cts = list()
    group_chID = dict()
    if len(range_list) != 2: raise RuntimeError("range should be length two")
    for chain in pose.chains:
        ct = 0
        for group in chain.groups:
            if range_list[0] <= group.groupNumber <= range_list[1]:
                atom = group.atom_by_name("CA")
                CA_coords[group.groupNumber] = atom.xyz
                ct += 1
                group_chID[group.groupNumber] = chain.ID
        chain_cts.append((chain.ID,ct))
    chain_cts = sorted(chain_cts, key=lambda x: x[1], reverse=True)
    if chain_cts[0][1] < (range_list[1] - range_list[0] + 1 - 5):
        return dict()
    elif chain_cts[0][1] == range_list[1] - range_list[0] + 1:
        return CA_coords
    else:
        ch = chain_cts[0][0]
        for groupN in list(CA_coords):
            if group_chID[groupN] != ch:
                try:
                    del CA_coords[groupN]
                except KeyError as ex:
                    print("No such key: '%s'" % ex.message)
        return CA_coords
    

def get_COM(CA_coords: typing.Dict[int,np.ndarray]) -> float:
    """ quick get COM for CA_coords
    """
    com = np.array((0.0,0.0,0.0))
    n = 0
    for xyz in CA_coords.values():
        com += xyz
        n += 1
    com /= n
    return com


def id_largest_rmsd_region(p1: daimyo.core.pose.Pose,
                           p2: daimyo.core.pose.Pose,
                           MAX_SIZE: int = 100,
                           WINDOW_SIZE: int = 50) -> typing.List[int]:
    """ between two poses, scans for windows of residues that have
        larger RMSD than the overall RMSD for structure
        then checks if far away pieces (at least 40A) also exist
        TODO - this doesn't catch regions where one small piece of a region 
        is moving a lot and then jacking up our region RMSD score...
    """
    SLIDING_RES = 5
    MIN_WINDOW_SIZE = 30
    EXTRA_RMSD = 1.0
    MIN_DIST = 40.0
    WINDOW_DEC = 10.0
    total_res = get_number_of_residues(p1)
    full_rmsd = rmsd_CA_noalign(get_CA_coords(p1),get_CA_coords(p2))
    # TODO - best cutoff here? 
    MIN_RMSD = full_rmsd + EXTRA_RMSD if full_rmsd + EXTRA_RMSD >= 2.0 else 2.0
    p1 = renumber_pose(p1)
    p2 = renumber_pose(p2)
    ## these starts and ends are INCLUSIVE
    starts = [x for x in range(1, total_res-WINDOW_SIZE+2, SLIDING_RES)] ## +2 ensures we get the last window
    ends = [x for x in range(WINDOW_SIZE, total_res+1, SLIDING_RES)]
    if ends[-1] != total_res: ends[-1] = total_res # add last bit of residues to the final group if needed
    possible_segs = dict()
    for (start, end) in zip(starts,ends):
        p1_CAs = get_specific_CAs(p1, (start,end))
        p2_CAs = get_specific_CAs(p2, (start,end))
        if not p1_CAs or not p2_CAs: continue
        seg_rmsd = rmsd_CA_noalign(p1_CAs, p2_CAs)
        if seg_rmsd > MIN_RMSD and max(p1_CAs.keys()) - min(p1_CAs.keys()) >= MIN_WINDOW_SIZE:     
            possible_segs[(min(p1_CAs.keys()), max(p1_CAs.keys()))] = seg_rmsd
    if len(possible_segs) == 0 and WINDOW_SIZE - WINDOW_DEC < MIN_WINDOW_SIZE:
        return None
    elif len(possible_segs) == 0 and WINDOW_SIZE - WINDOW_DEC >= MIN_WINDOW_SIZE:
        return id_largest_rmsd_region(p1, p2, MAX_SIZE=100, 
                WINDOW_SIZE=WINDOW_SIZE - WINDOW_DEC)
    else:
        # picking nonoverlap segs with highest rmsd 
        # consider making a seperate function
        # possible_segs is a dict of [(start, stop), rmsd]
        possible_segs = sorted(possible_segs.items(), key=itemgetter(1), reverse=True) # first sort by rmsd descending
        nonoverlap_segs = list()
        for seg in possible_segs:
            current_seg = seg[0]
            if len(nonoverlap_segs) == 0:
                nonoverlap_segs.append(current_seg)
            else:
                # we want to add ones with < 1/4 segment overlap
                if all(abs(current_seg[0] - prev_segs[0]) > WINDOW_SIZE*0.75 for prev_segs in nonoverlap_segs):
                    nonoverlap_segs.append(current_seg)
        saved_segs = list()
        if len(nonoverlap_segs) == 1:
            for seg in nonoverlap_segs:
                saved_segs.extend(list(seg))
        else:
            # getting things that are distant from the best rmsd
            # consider seperate function
            dist_COMs = dict()
            for seg1 in nonoverlap_segs:
                for seg2 in nonoverlap_segs:
                    if seg1[0] == seg2[0]: continue         # skip calculation for same
                    s1_CAs = get_specific_CAs(p1, seg1)
                    s2_CAs = get_specific_CAs(p1, seg2)
                    dist = np.linalg.norm(get_COM(s1_CAs) - get_COM(s2_CAs))
                    if dist in dist_COMs.values(): continue # skip adding if already there
                    dist_COMs[seg1,seg2] = dist
            # i want it to be a single list to interface properly with the deletion script
            saved_segs.extend(list(nonoverlap_segs[0]))
            # check if there is something far away that also moved:
            segs = [seg for seg, dist in dist_COMs.items() if saved_segs[0] in seg and dist > MIN_DIST]  # TODO - smaller dist??
            while (len(saved_segs)/2 < MAX_SIZE/WINDOW_SIZE) and len(segs) != 0:
                for seg in segs:
                    saved_segs.extend(list(seg))   
        return saved_segs
        

def commandline_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('general arguments')
    required.add_argument('-m', '--mode', type=str, required=True,
        choices=['pick_pdbs', 'rechain','find_region', "renumber", "energy_table"],
        help="mode to use")
    pick = parser.add_argument_group('arguments to pick top pdbs')
    pick.add_argument('-cd', '--containing_dir', type=str, default=None,
        help="dir containing pdbs")
    pick.add_argument('-od', '--out_dir', type=str, default="top_pdbs",
        help="name of out dir")
    pick.add_argument('-pn', '--pdb_name', type=str, default="",
        help="basename to use for pdbs (comes before _00##.pdb),"\
            "if more than one group in your directory")
    pick.add_argument('-st', '--score_type', type=str, default="",
        help="score type to sort pdbs on, default is total")
    pick.add_argument('-n', '--topn', type=int, default=5,
        help="top n models to take")
    rech = parser.add_argument_group('arguments to rechain/find_region')
    rech.add_argument('-s', '--pdbs', type=str, default="", nargs='+',
        help="pdb(s) to rechain")
    rech.add_argument('-o', '--outname', type=str, default="out",
        help="out name for output pdbs (only works for rechain)")
    args = parser.parse_args()

    if args.mode == "pick_pdbs":
        models = pick_topn_pdbs(args.containing_dir, 
                                out_directory=args.out_dir,
                                pdb_basename=args.pdb_name,
                                score_type=args.score_type,
                                n=args.topn)
        print(models)
    elif args.mode == "rechain":
        for pdb in args.pdbs:
            rename_chains(pdb)
    elif args.mode == "find_region":
        with open(args.pdbs[0], 'r') as pdbf:
            pose1 = read_pdb_lines(pdbf.read().split('\n'))
        with open(args.pdbs[1], 'r') as pdbf:
            pose2 = read_pdb_lines(pdbf.read().split('\n'))
        print(id_largest_rmsd_region(pose1,pose2))
    elif args.mode == "renumber":
        for pdb in args.pdbs:
            with open(pdb, 'r') as f:
                pose = read_pdb_lines(f.read().split('\n'))
            new_pose = renumber_pose(pose)
            new_pdb = pdb[:-4]+"_ren.pdb"
            write_pdb_file(new_pose, new_pdb)
    # for debugging
    elif args.mode == "energy_table":
        for pdb in args.pdbs:
            print(extract_energy_table(pdb))


def vis_stat_bfac_pdb(p: 'daimyo.core.pose', 
                  data: pd.DataFrame,
                  coln: str,
                  out_file: str) -> None:
    """ write a pdb where bfactors are changed to 1 or 0
        based on dict
        idk if this should really be here, it's not super usable 
        in other shit
    """
    for i, chain in enumerate(p.chains):
        for j, group in enumerate(chain.groups):
            key = f"{group.groupNumber}.{chain.ID}"
            for n in range(len(group.atoms)):
                p.chains[i].groups[j].atoms[n].bfactor = data.at[key, coln]
    write_pdb_file(p, out_file)


if __name__ == "__main__":
    commandline_main()
    
