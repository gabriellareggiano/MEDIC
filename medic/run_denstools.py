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
    parser.add_argument('-bc', '--binary_csv', help='csv of binary classification', type=str, required=False)
    parser.add_argument('--out_prefix', help='out prefix for csv', default="")
#    parser.add_argument('--plot', help='make a bunch of image plots', default=False, action='store_true')
    parser.add_argument('--use_Bs', help='use bfactors in density correlation calculation', default=False, action='store_true')
    parser.add_argument('--ray', help='run rays OG protocol', default=False, action='store_true')
    parser.add_argument('--atom_mask', required=False, type=float)
    args = parser.parse_args()
    return args


def run_denstools_perres(pdb: str,
                 mapfile: str,
                 reso: float,
                 bfacs: bool = False,
                 rays: bool = False,
                 atom_mask: float = None,
                 new_dens_zscore_flag: bool = False) -> pd.DataFrame:
    """ calculate ray's zscores for every residue and put in dataframe
    """
    rosetta_exe = os.path.join(os.getenv("ROSETTA3","/home/reggiano/bin/Rosetta/main"),
                    "source/bin/density_tools.default.linuxgccrelease")
    rosetta_db = os.path.join(os.getenv("ROSETTA3", "/home/reggiano/bin/Rosetta/main"),
                        "database")
    cmd = (f"{rosetta_exe} -database {rosetta_db}"
        f" -edensity::mapfile {mapfile}"
        f" -edensity::mapreso {reso}"
        f" -s {pdb}")
    if bfacs: cmd += " -bfacs"
    if rays: cmd += " -ray_z"
    if atom_mask: 
        cmd += f" -edensity::atom_mask {atom_mask:02f}"
        cmd += f" -edensity::atom_mask_min {atom_mask:.02f}"
    if new_dens_zscore_flag:
        cmd += " -density_zscores"
    else:
        cmd += " -perres"
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


def plot_zscore_vs_rmsd(df, col_name, pdb_fn) -> None:
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(8,8))
    plt.xlabel('rmsd')
    plt.ylabel('zscore')
    plt.title(f'{col_name} vs rmsd')
    y_max = ceil(df[col_name].max())
    y_min = floor(df[col_name].min())
    plt.ylim(y_min,y_max)
    plt.axvline(x=1, linewidth=3, linestyle='--', color='gray')
    plt.scatter(df['rmsd'],df[col_name], alpha=0.5, linewidth=1.5)

    out = f"{os.path.basename(pdb_fn)[:-4]}_{col_name}_rmsd.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")


## TODO this is the jankiest shit right now, i do not have time to deal with this
def commandline_main():
    args = parseargs()

    pdb1_scores = run_denstools_perres(args.pdbs[0], args.mapfile, args.reso, args.use_Bs, 
                               rays=args.ray, atom_mask=args.atom_mask)
    if len(args.pdbs) > 1:
        pdb2_scores = run_denstools_perres(args.pdbs[1], args.mapfile, args.reso, args.use_Bs, 
                                   rays=args.ray, atom_mask=args.atom_mask)

    if args.binary_csv:
        bin_classes = pd.read_csv(args.binary_csv)
        bin_classes = bin_classes.iloc[:,1:]
        pdb1_scores = pd.concat([pdb1_scores, bin_classes], axis=1)
        if len(args.pdbs) > 1:
            pdb2_scores = pd.concat([pdb2_scores, bin_classes], axis=1)

    if args.csv:
        pdb1_scores.to_csv(f"{args.out_prefix}zscores_{os.path.basename(args.pdbs[0])[:-4]}.csv")
        if len(args.pdbs) > 1:
            pdb2_scores.to_csv(f"{args.out_prefix}zscores_{os.path.basename(args.pdbs[1])[:-4]}.csv")


if __name__ == "__main__":
    commandline_main()
