#!/usr/bin/env python

import pandas as pd
import argparse
import pyrosetta


def run(pdbf: str, mapf: str, reso: float) -> pd.DataFrame:
    """
    use pyrosetta to calculate density scores
    use setup for dens boolean if density scoring not done previously on pose
    """
    atom_mask = 3.2
    flags = ['-mute all',
            '-ignore_unrecognized_res',
            '-default_max_cycles 200',
            f'-edensity::mapfile {mapf}',
            f'-edensity::mapreso {reso}']
    pyrosetta.init(' '.join(flags))
    pose = pyrosetta.pose_from_file(pdbf)

    # set the atom mask
    emmap = pyrosetta.rosetta.core.scoring.electron_density.getDensityMap()
    emmap.setAtomMask(atom_mask)

    setupdens_mover = pyrosetta.rosetta.protocols.electron_density.SetupForDensityScoringMover()
    dens_zsc_mover = pyrosetta.rosetta.protocols.electron_density.DensityZscores()

    setupdens_mover.apply(pose)
    dens_zsc_mover.apply(pose)
    data = {"resn":[],
            "chID":[],
            "resi":[],
            "bfactor":[],
            "local_bfactor":[],
            "perResCC":[],
            "per3ResCC":[],
            "perResZDensWin1":[],
            "perResZDensWin3":[]}
    pdbinfo = pose.pdb_info()
    for i in range(1,pose.total_residue()+1):
        data["resn"].append(pose.residue(i).name3())
        data["chID"].append(pdbinfo.chain(i))
        data["resi"].append(pdbinfo.number(i))
    data["bfactor"] = dens_zsc_mover.get_res_bfacs()
    data["local_bfactor"] = dens_zsc_mover.get_nbrhood_bfacs()
    data["perResCC"] = dens_zsc_mover.get_win1_denscc()
    data["per3ResCC"] = dens_zsc_mover.get_win3_denscc()
    data["perResZDensWin1"] = dens_zsc_mover.get_win1_dens_zscore()
    data["perResZDensWin3"] = dens_zsc_mover.get_win3_dens_zscore()
    data = pd.DataFrame.from_dict(data)
    # drop virtual residues
    data.drop(data[data['resn'].str.match('XXX')].index, inplace=True)
    return data


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pdb', help='pdbs to calculate', type=str, required=True, nargs="+")
    parser.add_argument('-m', '--mapfile', help='density map', type=str, required=True)
    parser.add_argument('-r', '--reso', help='map resolution', type=float, required=True)
    parser.add_argument('--outfile', help='out prefix for csv', default="dens_zscores.csv")
    args = parser.parse_args()
    return args


def commandline_main():
    args = parseargs()
    pdb_scores = run(args.pdb, args.mapfile, args.reso)
    pdb_scores.to_csv(args.outfile)


if __name__ == "__main__":
    commandline_main()
