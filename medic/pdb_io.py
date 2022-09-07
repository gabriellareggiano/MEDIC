#!/usr/bin/env python

import sys

import numpy as np

from medic import pose, util


def fon(s):
    """Float or None

    Args:
        s: A string to convert
    Returns:
        Float if can be floated, else returns None
    """
    try:
        return float(s)
    except ValueError:
        return None


def ion(s):
    """Int or None

    Args:
        s: A string to convert
    Returns:
        Int if can be floated, else returns None
    """
    try:
        return int(s)
    except ValueError:
        return None


def pad_atom_name(atom_name_raw):
    atom_name = atom_name_raw.strip()
    atom_name_len = len(atom_name)
    if atom_name_len == 4:
        return atom_name
    elif atom_name_len == 3:
        if not atom_name[0].isalpha():
            return f"{atom_name} "
        else:
            return f" {atom_name}"
    elif atom_name_len == 2:
        return f" {atom_name} "
    elif atom_name_len == 1:
        return f" {atom_name}  "
    else:
        raise RuntimeError(f"could not pad >{atom_name}< from >{atom_name_raw}<")


def atom_from_pdb_line(line):
    line = line.strip("\n")
    if len(line) != 80:
        line = line + " " * (len(line) - 80 + 1)
    atom_name = line[12:16].strip()
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    occupancy = fon(line[55:60].strip())
    t_factor = fon(line[60:66].strip())
    element_symbol = line[77:78].strip()
    charge = line[79:80]
    atm = pose.Atom(
        np.array((x, y, z)),
        atomName=atom_name,
        element=element_symbol,
        occupancy=occupancy,
        bfactor=t_factor,
        charge=charge,
    )
    return atm


def group_and_one_atom_from_line(line):
    atm = atom_from_pdb_line(line)
    groupName = line[17:20].strip()
    if line.startswith("ATOM"):
        chemCompType = "PEPTIDE LINKING"
        singleLetter = util.AA321(groupName, False)
        if singleLetter is None:
            singleLetter = "X"
    else:
        chemCompType = "other"
        singleLetter = "X"
    group = pose.Group(
        groupName=groupName,
        groupNumber=int(line[22:26].strip()),
        atoms=[atm],
        chemCompType=chemCompType,
        singleLetter=singleLetter,
    )
    return group


def read_pdb_lines(lines, get_het=False):
    lines = [line.strip() for line in lines]
    p = pose.Pose()
    chain = None
    group = None
    for i, line in enumerate(lines):
        if not line.startswith("ATOM") and not (line.startswith("HETATM") and get_het):
            continue
        if not chain and not group and len(p.chains) == 0:
            group = group_and_one_atom_from_line(line)
            chain = pose.Chain(ID=line[21:22])
        else:
            curr_chain_id = line[21:22]
            if chain.ID == curr_chain_id:
                curr_group_number = int(line[22:26].strip())
                if group.groupNumber == curr_group_number:
                    assert group.groupName == line[17:20].strip()
                    atm = atom_from_pdb_line(line)
                    group.atoms.append(atm)
                else:
                    if group.groupNumber == curr_group_number - 1:
                        chain.groups.append(group)
                        group = group_and_one_atom_from_line(line)
                    else:
                        chain.groups.append(group)
                        p.chains.append(chain)
                        group = group_and_one_atom_from_line(line)
                        chain = pose.Chain(ID=line[21:22])
            else:
                chain.groups.append(group)
                p.chains.append(chain)
                group = group_and_one_atom_from_line(line)
                chain = pose.Chain(ID=line[21:22])
    chain.groups.append(group)
    p.chains.append(chain)
    return p


def read_pdb_file(pdbfile, get_het=False):
    pdb_lines = open(pdbfile, "r").readlines()
    try:
        pose = read_pdb_lines(pdb_lines, get_het)
    except Exception:
        print(RuntimeError(f"Error parsing {pdbfile}"))
        raise
    return pose


def get_pdb_string(p):
    """Write to pdbstring

    Args:
        p: pose
    Returns:
        pdbstring: a string to write to a pdb file
    """
    ATOMS = ["PEPTIDE LINKING", "L-PEPTIDE LINKING"]
    pdbstring = ""
    atmnumber = 1
    for chain in p.chains:
        pdb_chainID = chain.ID[0]
        for group in chain.groups:
            atm_hetatm = "ATOM" if group.chemCompType.upper() in ATOMS else "HETATM"
            for atm in group.atoms:
                # Set non required
                occupancy = f"{' ':6.2s}" if not atm.occupancy else f"{atm.occupancy:6.2f}"
                bfactor = f"{' ':6.2s}" if not atm.bfactor else f"{atm.bfactor:6.2f}"
                # Always set charge empty - no one uses it!
                charge = "  "
                atom_name = pad_atom_name(atm.atomName)
                pdbstring += (
                    f"{atm_hetatm:6s}{atmnumber:5d} {atom_name} {group.groupName:3s} {pdb_chainID:1s}"
                    f"{group.groupNumber:4d}    {atm.xyz[0]:8.3f}{atm.xyz[1]:8.3f}{atm.xyz[2]:8.3f}"
                    f"{occupancy}{bfactor}          {atm.element:>2s}{charge}\n"
                )
                atmnumber += 1
    return pdbstring


def write_pdb_file(p, filename):
    """Write to pdb file

    Args:
        p: pose
        filename: file to write to
    """
    try:
        f = open(filename, "w")
    except IOError:
        print(f"Could not open >{filename}< for writing", file=sys.stderr)
        sys.exit(1)
    pdb_string = get_pdb_string(p)
    f.write(pdb_string)
    f.close()
