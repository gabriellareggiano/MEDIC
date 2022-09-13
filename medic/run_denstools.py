#!/usr/bin/env python

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import subprocess
import shlex
import io
import argparse
import os
from math import ceil, floor
mpl.use('Agg')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pdbs', help='pdbs to calculate', type=str, required=True, nargs="+")
    parser.add_argument('--csv', help='save a csv?', action="store_true", default=False)
    parser.add_argument('-m', '--mapfile', help='density map', type=str, required=True)
    parser.add_argument('-r', '--reso', help='map resolution', type=float, required=True)
    parser.add_argument('--out_prefix', help='out prefix for csv', default="")
    args = parser.parse_args()
    return args


def run_denstools_perres(pdb: str,
                 mapfile: str,
                 reso: float) -> pd.DataFrame:
    """ calculate ray's zscores for every residue and put in dataframe
    """
    ATOM_MASK = 3.2
    rosetta_exe = os.path.join(os.getenv("ROSETTA3","/home/reggiano/bin/Rosetta/main"),
                    "source/bin/density_tools.default.linuxgccrelease")
    rosetta_db = os.path.join(os.getenv("ROSETTA3", "/home/reggiano/bin/Rosetta/main"),
                        "database")
    cmd = (f"{rosetta_exe} -database {rosetta_db}"
        f" -edensity::mapfile {mapfile}"
        f" -edensity::mapreso {reso}"
        f" -s {pdb}"
        f" -edensity::atom_mask {ATOM_MASK:02f}"
        f" -edensity::atom_mask_min {ATOM_MASK:.02f}"
        " -density_zscores")
    print(cmd)
    ret = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode:
        raise RuntimeError(f"dens tools failed\n {cmd}")
    out = io.StringIO()
    for line in ret.stdout.decode().split('\n'):
        if "density_tools" in line and "pdb" not in line and "BFACTORS" not in line:
            out.write(line+'\n')
    out.seek(0)
    df = pd.read_csv(out, sep="\s+", header=0)
    out.close()
    return df.drop(['density_tools:', 'PERRESCC', 'residue'], axis=1)


def commandline_main():
    args = parseargs()
    pdb1_scores = run_denstools_perres(args.pdbs[0], args.mapfile, args.reso)
    if args.csv:
        pdb1_scores.to_csv(f"{args.out_prefix}zscores_{os.path.basename(args.pdbs[0])[:-4]}.csv")


if __name__ == "__main__":
    commandline_main()
