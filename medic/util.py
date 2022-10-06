"""Simple core utils
"""
import pandas as pd
import medic.pose
import os
import glob


def get_number_of_residues(pose: medic.pose.Pose) -> int:
    n = 0
    for chain in pose.chains:
        for group in chain.groups:
            n += 1
    return n


def AA321(residue, toraise):
    dic = {
        "GLY": "G",
        "ALA": "A",
        "VAL": "V",
        "LEU": "L",
        "ILE": "I",
        "PRO": "P",
        "CYS": "C",
        "MET": "M",
        "HIS": "H",
        "PHE": "F",
        "TYR": "Y",
        "TRP": "W",
        "ASN": "N",
        "GLN": "Q",
        "SER": "S",
        "THR": "T",
        "LYS": "K",
        "ARG": "R",
        "ASP": "D",
        "GLU": "E",
        "5HP": "Q",
        "ABA": "C",
        "AGM": "R",
        "CEA": "C",
        "CGU": "E",
        "CME": "C",
        "CSB": "C",
        "CSE": "C",
        "CSD": "C",
        "CSO": "C",
        "CSP": "C",
        "CSS": "C",
        "CSW": "C",
        "CSX": "C",
        "CXM": "M",
        "CYM": "C",
        "CYG": "C",
        "DOH": "D",
        "FME": "M",
        "GL3": "G",
        "HYP": "P",
        "KCX": "K",
        "LLP": "K",
        "LYZ": "K",
        "MEN": "N",
        "MGN": "Q",
        "MHS": "H",
        "MIS": "S",
        "MLY": "K",
        "MSE": "M",
        "NEP": "H",
        "OCS": "C",
        "PCA": "Q",
        "PTR": "Y",
        "SAC": "S",
        "SEP": "S",
        "SMC": "C",
        "STY": "Y",
        "SVA": "S",
        "TPO": "T",
        "TPQ": "Y",
        "TRN": "W",
        "TRO": "W",
        "YOF": "Y",
        "AGLN": "Q",
        "AASN": "N",
        "AVAL": "V",
        "AHIS": "H",
        "ASER": "S",
        "ATHR": "T",
        "MLZ": "K",
    }
    if toraise:
        if residue not in dic:
            raise ValueError(
                f"Could not map the residue ==>{residue}" "<== in our three letter to one letter AA dictionary"
            )
        else:
            return dic[residue]
    else:
        if residue not in dic:
            return None
        else:
            return dic[residue]


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


def clean_dan_files(input_file_name):
    tmp_files = glob.glob(f"{input_file_name[:-4]}*_r[0-9][0-9][0-9][0-9].???")
    for f in tmp_files:
        try:
            if os.path.isfile(f):
                os.remove(f)
        except:
            print("Failed to clean for", f)


def clean_pdb(pdbf):
    compatible_res = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU",
             "GLN", "GLY", "HIS", "ILE", "LEU", "LYS",
             "MET", "PHE", "PRO", "SER", "THR", "TRP",
             "TYR", "VAL"]
    with open(pdbf, 'r') as f:
        pdb_lines = [line.strip() for line in f.readlines()
                    if line[:4] == "ATOM" and line[17:21].strip() in compatible_res]
    new_pdb = pdbf[:-4]+"_clean.pdb"
    with open(new_pdb, 'w') as f:
        f.write('\n'.join(pdb_lines))
    return new_pdb
    