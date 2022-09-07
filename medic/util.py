"""Simple core utils
"""

import typing
import os
import sys
import asyncio
import math
from typing import List

import requests
import numpy as np
import mmtf

import daimyo
import daimyo.core.pose
from daimyo.core.sequence import Sequence
from daimyo.core.sequence_alignment import SmithWaterman as SW


def iter_pose_groups(pose):
    for chain in pose.chains:
        for group in chain.groups:
            yield group


def vdw_radius_from_element(ele: str):
    D = {"H": 1.1, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}
    return D[ele]


async def run_async_subp(task: str) -> int:
    """Simple wrapper to send job to be processed
    """
    p = await asyncio.create_subprocess_shell(task)
    ret = await p.wait()
    if ret != 0:
        print("Return from task failed, quitting.")
        print(f"Failed task: {task}")
        raise RuntimeError(f"Failed task: {task}")
    return ret


def get_idx_dict_from_line(line):
    """Map split indicies based on score name

    Input:
        line: A SCORE: line
    Returns:
        idx_dict: A dictionary mapping "score" to idx #
    """
    idx_dict = {}
    line_sp = line.split()[1:]
    for i, x in enumerate(line_sp):
        idx_dict[x] = i + 1
    return idx_dict


def backup_is_hetatm(groupName: str) -> bool:
    """Get is_hetatm based on

    Args:
        group: a mmtf group
    return:
        Bool based on if we think it is HETATM or not
    """
    known_HETs = ["GTP", "GDP", "CTP", "CDP", "ATP", "ADP", "TTP", "TDP", "HOH", "MG", "CL", "FE"]
    if groupName in known_HETs:
        return True
    cAAs = [
        "GLY",
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "PRO",
        "CYS",
        "MET",
        "HIS",
        "PHE",
        "TYR",
        "TRP",
        "ASN",
        "GLN",
        "SER",
        "THR",
        "LYS",
        "ARG",
        "ASP",
        "GLU",
    ]
    if groupName in cAAs:
        return False
    # We don't know, keep unless we learn otherwise
    return False


def is_hetatm(group):
    hetatm_types = [
        "D-BETA-PEPTIDE, C-GAMMA LINKING",
        "D-GAMMA-PEPTIDE, C-DELTA LINKING",
        "D-PEPTIDE COOH CARBOXY TERMINUS",
        "D-PEPTIDE LINKING",
        "D-PEPTIDE NH3 AMINO TERMINUS",
        "D-SACCHARIDE",
        "D-SACCHARIDE 1,4 AND 1,4 LINKING",
        "D-SACCHARIDE 1,4 AND 1,6 LINKING",
        "DNA LINKING",
        "DNA OH 3 PRIME TERMINUS",
        "DNA OH 5 PRIME TERMINUS",
        "L-BETA-PEPTIDE, C-GAMMA LINKING",
        "L-DNA LINKING",
        "L-GAMMA-PEPTIDE, C-DELTA LINKING",
        "L-PEPTIDE COOH CARBOXY TERMINUS",
        # "L-PEPTIDE LINKING",
        "L-PEPTIDE NH3 AMINO TERMINUS",
        "L-RNA LINKING",
        "L-SACCHARIDE",
        "L-SACCHARIDE 1,4 AND 1,4 LINKING",
        "L-SACCHARIDE 1,4 AND 1,6 LINKING",
        "NON-POLYMER",
        "OTHER",
        # "PEPTIDE LINKING",
        "PEPTIDE-LIKE",
        "RNA LINKING",
        "RNA OH 3 PRIME TERMINUS",
        "RNA OH 5 PRIME TERMINUS",
        "SACCHARIDE",
        0,
    ]
    if group["chemCompType"] == "":  # Pymol model case
        return backup_is_hetatm(group["groupName"])
    elif group["chemCompType"] in hetatm_types:
        return True
    return False


def apply_rot_matrix_to_CA_coords(CA_coords, rot_matrix):
    for i, key in enumerate(CA_coords):
        CA_coords[i] = np.dot(key, rot_matrix)
    return CA_coords


def matrix_from_kabsch_align(r1_coords: np.ndarray, r2_coords: np.ndarray):
    """Perform kabsch align on two sets of coordinates

    Notes:
        We align two sets of numpy coordinates (indicies must match, ie you must want
        idx 0 of 'r1_coords1' and 'r2_coords' to align together).
    """
    E0 = np.sum(np.sum(r1_coords * r1_coords, axis=0), axis=0) + np.sum(np.sum(r2_coords * r2_coords, axis=0), axis=0)
    V, S, Wt = np.linalg.svd(np.dot(np.transpose(r2_coords), r1_coords))
    reflect = float(str(float(np.linalg.det(V) * np.linalg.det(Wt))))
    if np.isclose(-1.0, reflect):
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    rotation_matrix = np.dot(V, Wt)
    RMSD = E0 - (2.0 * sum(S))
    RMSD = np.sqrt(abs(RMSD / len(r1_coords)))
    return rotation_matrix, RMSD


def get_chain_rawpdb(pdbid: str, chain: str = None, idx_at_0: bool = True, keep_het: bool = True) -> str:
    """Use mmtf to build a nice pdb string

    Args:
        pdbid: pdb id
        chain: desired chain name, None=get all chains
        idx_at_0: Determine atom idx from this builder not the pdb
        keep_het: keep HETATM in pdbstring
        model: get particular model, None == Get all models
    return
        pdb chain text
    """
    pdb_lines = []
    try:
        if os.environ["mmtfdb"]:
            pdb_data = mmtf.parse(f"{os.environ['mmtfdb']}/{pdbid.lower()}.mmtf")
        else:
            pdb_data = mmtf.fetch(pdbid.upper())
    except Exception:  # TODO find out what exception this is
        print(f"Failure to obtain (mmtf)pdb: {pdbid}")
        sys.exit(-1)
    chainIndex = 0
    groupIndex = 0
    atomIndex = 0
    atom_count = 0
    current_line = ""
    # for a in range(pdb_data.num_models): Keep :D
    for a in range(1):
        for b in range(pdb_data.chains_per_model[a]):
            for c in range(pdb_data.groups_per_chain[chainIndex]):
                groupType = pdb_data.group_type_list[groupIndex]
                group = pdb_data.group_list[groupType]
                for d, _ in enumerate(group["atomNameList"]):
                    if is_hetatm(group):
                        current_line += "HETATM"  # 0-6
                    else:
                        current_line += "ATOM  "  # 0-6
                    if idx_at_0:
                        current_line += f"{pdb_data.atom_id_list[atomIndex]:5d}"  # 6:11
                    else:
                        current_line += f"{atom_count:5d}"  # 6:11
                    current_line += f" {group['atomNameList'][d]:^4s}"  # 11:16
                    current_line += f" {group['groupName']:3s} "  # 16:20
                    current_line += f"{pdb_data.chain_name_list[chainIndex].strip():1s}"  # 20:22
                    current_line += f"{pdb_data.group_id_list[groupIndex]:4d}"  # 22:26
                    current_line += "    "
                    current_line += f"{pdb_data.x_coord_list[atomIndex]:8.3f}"
                    current_line += f"{pdb_data.y_coord_list[atomIndex]:8.3f}"
                    current_line += f"{pdb_data.z_coord_list[atomIndex]:8.3f}"
                    current_line += f"{pdb_data.occupancy_list[atomIndex]:6.2f}"
                    current_line += " " * 16
                    current_line += f"{group['elementList'][d]:>2s}"
                    if pdb_data.chain_name_list[chainIndex] == chain and chain:
                        atom_count += 1
                        pdb_lines.append(current_line)
                    elif not chain:
                        atom_count += 1
                        pdb_lines.append(current_line)
                    current_line = ""
                    atomIndex += 1
                groupIndex += 1
            chainIndex += 1
    if not keep_het:
        pdb_lines = [line for line in pdb_lines if line.startswith("ATOM")]
    return "\n".join(pdb_lines)


def import_fasta_from_string(fasta_string: str) -> typing.List[Sequence]:
    """parse a fasta string to a fasta object

    :param fasta_string: a read fasta string
    :type fasta_string: str

    :return A list of Sequence objects
    """
    fastalist = []
    name = ""
    sequence = ""
    for line in fasta_string.split("\n"):
        if not line:
            continue
        elif line.startswith(">"):
            if len(sequence) != 0:
                fastalist.append(Sequence(name, sequence, None))
            name = line[1:].strip()
            sequence = ""
        else:
            sequence += line.strip("\n")
    # last sequence in file
    else:
        if len(sequence) != 0:
            fastalist.append(Sequence(name, sequence, None))
    return fastalist


def import_fastafile(fastafile: str) -> typing.List[Sequence]:
    """
    A simple function to parse and import all fastas from a given file.
    No checks are made here (currently)

    Args:
        fastafile: A fasta file

    Returns:
        A list of Sequence objects
    """
    return import_fasta_from_string(open(fastafile).read())


def import_fastas_from_files(filenames: typing.List[str]):
    """
    Load fastas into a list, and returns it (only grabs files that end in .fasta)
    that maps the name of the fasta input (at the end of the '>') to the fasta

    It will throw an error if we have two fastas with the same input name.
    But can handle multiple fastas in one file.
    Also will rename names to be at least 6 characters, but max 10, if you
    didn't preformat it (required)

    TODO: need to setup parents if they are pre-known
    """
    fastas = []
    for f in filenames:
        fastas += import_fastafile(f)
    fasta_names = [x.name for x in fastas]
    if len(fasta_names) != len(set(fasta_names)):
        print("Found fastas with duplicate names! Please check your inputs.")
        print(fasta_names)
        print("List with no duplicates:")
        print(set(fasta_names))
        raise RuntimeError
    return fastas


def sequences_from_pose_chains(pose: daimyo.core.pose.Pose, fail: bool = False) -> List[str]:
    """Get sequence object from pose

    :param pose: pose object
    :type pose: daimyo.core.pose.Pose
    :param fail: should we fail if we can't make the pose object
    :type bool
    """
    seqs = []
    for chain in pose.chains:
        seqs.append("")
        for group in chain.groups:
            seqs[-1] += group.singleLetter

    return seqs


def sequence_from_pose(pose: daimyo.core.pose.Pose) -> Sequence:
    """Get sequence object from pose

    :param pose: pose object
    :type pose: daimyo.core.pose.Pose
    :param fail: should we fail if we can't make the pose object
    :type bool
    """
    seq = ""
    for chain in pose.chains:
        for group in chain.groups:
            seq += group.singleLetter
    return Sequence(name=pose.ID, sequence=seq)


# SKEDDTLRRFRYLLGLTDLFRHFIETNPNPKIREIMKEIDRQNEEEARQRKRGGRQGGATSERRRRTEAEEDAELLKDEKDGGSAETVFRESPPFIQGTMRDYQIAGLNWLISLHENGISGILADEMGLGKTLQTIAFLGYLRHIMGITGPHLVTVPKSTLDNWKREFEKWTPEVNVLVLQGAKEERHQLINDRLVDENFDVCITSYEMILREKAHLKKFAWEYIIIDEA--------SLAQVIRM-FNSRNRLLITGTPLQNNLHELWALLNFLLPDVFGDSEAFDQWFSGQDRDQ-----------DTVVQQLHRVLRPFLLRRVKSDVEKSLLPKKEINVYIGMSEMQVKWYQKILEKDIDAVNGAGG---KRESKTRLLNIVMQLRKCCNHPYLFEGAEPG--------PPYTTDEHLIYNAGKMVVLDKLLKRIQKQGSRVLIFSQMSRLLDILEDYCVFRGYKYCRIDGSTAHEDRIAAIDEYNKPGSDKFIFLLTTRAGGLGINLTTADIVILYDSDWNPQADLQAMDRAHRIGQTKQVVVYRFVTDNAIEEKVLERAAQKLRLDQLVIQQGRAQVAAKAAANKDELLSMIQHGAEKVFQTKGAFG----------------------TMAEKGSQLDDDDIDAILQAGETRTKELNA
# -------------------------------------------------------------------------------------ETVFRESPPFIQGTMRDYQIAGLNWLISLHENGISGILADEMGLGKTLQTIAFLGYLRHIMGITGPHLVTVPKSTLDNWKREFEKWTPEVNVLVLQGAKEERHQLINDRLVDENFDVCITSYEMILREKAHLKKFAWEYIIIDEAHRIKNEESSLAQVIRM-FNSRNRLLITGTPLQNNLHELWALLNFLLPDVFGDSEAFDQWFSGQDRDQ-----------DTVVQQLHRVLRPFLLRRVKSDVEKSLLPKKEINVYIGMSEMQVKWYQKILEKDIDAVNGAGG---KRESKTRLLNIVMQLRKCCNHPYLFEGAEPG--------PPYTTDEHLIYNAGKMVVLDKLLKRIQKQGSRVLIFSQMSRLLDILEDYCVFRGYKYCRIDGSTAHEDRIAAIDEYNKPGSDKFIFLLTTRAGGLGINLTTADIVILYDSDWNPQADLQAMDRAHRIGQTKQVVVYRFVTDNAIEEKVLERAAQKLRLDQLVIQQGRAQVAAKAAANKDELLSMIQHGAEKVFQTKGAFG----------------------TMAEKGSQLDDDDIDAILQAGETRTKELNA
def trim_pose_to_alignment(
    pose: daimyo.core.pose.Pose, pose_gapped_alignment: str, target_gapped_alignment: str, delete_mismatch=False
) -> daimyo.core.pose.Pose:
    """trim a pose until it no longer has extra residues compared to the alignment

    :param pose: pose object
    :type pose: daimyo.core.pose.Pose
    :param pose_gapped_alignment: the pose sequence alignment
    :type pose_gapped_alignment: str
    :param target_gapped_alignment: the pose sequence alignment
    :type target_gapped_alignment: str
    :param delete_mismatch: should I delete mismatches?
    """
    assert len(pose_gapped_alignment) == len(target_gapped_alignment)
    to_delete = []
    pose_count = 0
    new_seq = list(pose_gapped_alignment)
    for i in range(len(pose_gapped_alignment)):
        # Unaligned region
        if target_gapped_alignment[i] == "-" and pose_gapped_alignment[i] != "-":
            to_delete.append(pose_count)
            new_seq[i] = "-"
        elif (
            delete_mismatch
            and target_gapped_alignment[i] != "-"
            and pose_gapped_alignment[i] != "-"
            and target_gapped_alignment[i] != pose_gapped_alignment[i]
        ):
            to_delete.append(pose_count)
        if pose_gapped_alignment[i] != "-":
            pose_count += 1

    pose_count = 0
    to_delete_chain_group_idxs = []
    for i, chain in enumerate(pose.chains):
        for j, group in enumerate(chain.groups):
            if pose_count in to_delete:
                to_delete_chain_group_idxs.append((i, j))
            pose_count += 1

    for chain_idx, group_idx in reversed(to_delete_chain_group_idxs):
        del pose.chains[chain_idx].groups[group_idx]
    return pose


def remove_unk_from_pose(p: daimyo.core.pose.Pose):
    to_delete = []
    for i, chain in enumerate(p.chains):
        for j, group in enumerate(chain.groups):
            if group.singleLetter == "X":
                to_delete.push_back((i, j))
    for c_idx, g_idx in reversed(to_delete):
        del p.chains[c_idx].groups[g_idx]
    return p


def real_fasta_from_pdb_and_chain(pdbID: str, chain: str) -> str:
    url = f"https://www.rcsb.org/pdb/rest/search/describeMol?structureId={pdbID}.{chain}"
    ret = requests.get(url).text
    print(ret)
    uniprot_id = None
    for line in ret.split("\n"):
        if "accession" in line:
            uniprot_id = line.split('"')[1]
    if not uniprot_id:
        return None
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    return requests.get(url).text


def canonical_321(single_letter):
    dic = {
        "G": "GLY",
        "A": "ALA",
        "V": "VAL",
        "L": "LEU",
        "I": "ILE",
        "P": "PRO",
        "C": "CYS",
        "M": "MET",
        "H": "HIS",
        "F": "PHE",
        "Y": "TYR",
        "W": "TRP",
        "N": "ASN",
        "Q": "GLN",
        "S": "SER",
        "T": "THR",
        "K": "LYS",
        "R": "ARG",
        "D": "ASP",
        "E": "GLU",
    }
    return dic[single_letter]


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


# ### PDB_INFO manipulations
def align_pose_chains_to_string(pose: daimyo.core.pose.Pose, seq: str):
    sw_align = SW(120, 0, -100, 0)
    pose_seqs = sequences_from_pose_chains(pose)

    for i, chain_sequence in enumerate(pose_seqs):
        target_aln, chain_seq_aln = sw_align.apply(seq, chain_sequence)
        assert len(seq) == len(target_aln)

        residue_numbers = []
        for j, x in enumerate(target_aln):
            if chain_seq_aln[j] != "-":
                residue_numbers.append(j + 1)

        for j, group in enumerate(pose.chains[i].groups):
            group.groupNumber = residue_numbers[j]
    return pose


def align_pose_chains_to_strings(
    pose: daimyo.core.pose.Pose,
    sequences: typing.List[str],
    chain_ids: typing.Optional[typing.List[str]],
    fail: bool = True,
) -> daimyo.core.pose.Pose:
    sw_align = SW(120, -999, -100, 0)
    pose_seqs = sequences_from_pose_chains(pose)

    for i, chain_sequence in enumerate(pose_seqs):
        correct_seq = None
        correct_seq_idx = 0
        for j, possible_seq in enumerate(sequences):
            print(i, chain_sequence)
            print(j, possible_seq)
            try:
                target_aln, chain_seq_aln = sw_align.apply(possible_seq, chain_sequence)
                print(target_aln)
                print(chain_seq_aln)
            except RuntimeError:
                if fail:
                    raise
                else:
                    continue
            if "-" not in target_aln and chain_sequence in chain_seq_aln:
                correct_seq = possible_seq
                correct_seq_idx = j
                break
        if not correct_seq:
            if fail:
                print(chain_sequence)
                raise RuntimeError("missing seq")
            else:
                continue

        residue_numbers = []
        for j, x in enumerate(target_aln):
            if chain_seq_aln[j] != "-":
                residue_numbers.append(j + 1)

        for j, group in enumerate(pose.chains[i].groups):
            group.groupNumber = residue_numbers[j]
        if chain_ids:
            pose.chains[i].name = chain_ids[correct_seq_idx]
            pose.chains[i].ID = chain_ids[correct_seq_idx]
    return pose


def align_pose_to_string(pose: daimyo.core.pose.Pose, seq: str):
    sw_align = SW(120, 0, -100, 0)
    pose_seq = sequence_from_pose(pose)
    pose_aln, target_aln = sw_align.apply(pose_seq.sequence, seq)
    if len(seq) != len(target_aln):
        raise RuntimeError(f"\nt: {target_aln}\np: {pose_aln}")
    residue_numbers = []
    for i, x in enumerate(target_aln):
        if pose_aln[i] != "-":
            residue_numbers.append(i + 1)
    count = 0
    for chain in pose.chains:
        for group in chain.groups:
            group.groupNumber = residue_numbers[count]
            count += 1
    return pose


def align_pose_to_sequence(pose: daimyo.core.pose.Pose, seq: daimyo.core.sequence.Sequence):
    return align_pose_to_string(pose, seq.sequence)


def rechain_pose(pose: daimyo.core.pose.Pose, chain_name: str = "", chain_id: str = ""):
    for chain in pose.chains:
        if chain_name:
            chain.name = chain_name
        if chain_id:
            chain.ID = chain_id
    return pose


# ### simple string manipulations


def rechain_pdb_string(pdb_string: str, chain: str) -> str:
    """Take a pdb string, and replace the chain for all atoms to {chain}

    :param pdb_string: a pdb string with newlines
    :param chain: a char that will be the new string
    """
    split_string = pdb_string.split("\n")

    for i, line in enumerate(split_string):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            split_string[i] = line[:21] + chain + line[22:]
    return "\n".join(split_string)


# ##### CA_pose functions #####


def get_CA_coords(pose: daimyo.core.pose.Pose, force_update: bool = False) -> typing.Dict[int, np.ndarray]:
    """pose to residue number: CA_coordinate dict

    """
    CA_coords = {}
    for chain in pose.chains:
        for group in chain.groups:
            for atom in group.atoms:
                if atom.atomName.strip() == "CA":
                    CA_coords[group.groupNumber] = atom.xyz
    pose.CA_coords = CA_coords
    return CA_coords


def get_chained_CA_coords(pose):
    chained_coords = {}
    for chain in pose.chains:
        for group in chain.groups:
            for atom in group.atoms:
                if atom.atomName.strip() == "CA":
                    chained_coords[f"{chain.name}_{group.groupNumber}"] = atom.xyz
    return chained_coords


def recenter_coords_at_point(coords: np.ndarray, COM: np.ndarray) -> np.ndarray:
    """Recenter the coordinates of a pose at a point
    """
    return coords - np.array(COM)


def trim_target_and_mobile(target, mobile):
    trimmed_target_coords = []
    trimmed_mobile_coords = []
    for key, val in target.items():
        if key in mobile:
            trimmed_target_coords.append(val)
            trimmed_mobile_coords.append(mobile[key])
    return np.array(trimmed_target_coords), np.array(trimmed_mobile_coords)


def aligned_rmsd_from_CAs(
    all_target_CAs: typing.Dict[int, np.ndarray], all_mobile_CAs: typing.Dict[int, np.ndarray]
) -> float:
    target_CAs, mobile_CAs = trim_target_and_mobile(all_target_CAs, all_mobile_CAs)
    target_CAs = recenter_coords_at_point(target_CAs, target_CAs.mean(axis=0))
    mobile_CAs = recenter_coords_at_point(mobile_CAs, mobile_CAs.mean(axis=0))
    if not len(target_CAs) or not len(mobile_CAs):
        return 100
    _, RMSD = matrix_from_kabsch_align(target_CAs, mobile_CAs)
    return RMSD


def get_neighbors_from_CAs(
    all_CAs: typing.Dict[int, np.ndarray], max_seq_d: int = 12, min_seq_d: int = 12, max_d_d: float = 0
) -> typing.List[typing.Tuple[int, np.ndarray, np.ndarray]]:
    """
        Get neighbors which is a map for each residue_num to the residues+their xyzs in the current neighborhood
        result:
            [ ( parter_residue_num, parter_xyz, partner_evaluated_distance), ...]
    """
    neighbors_map = {}
    for rsd_num_i, xyz_i in all_CAs.items():
        result = []
        for rsd_num_j, xyz_j in all_CAs.items():
            if rsd_num_j == rsd_num_i:
                continue
            seq_d = abs(rsd_num_i - rsd_num_j)
            if max_seq_d and seq_d > max_seq_d:
                continue
            if min_seq_d and seq_d < min_seq_d:
                continue
            d = np.linalg.norm(xyz_i - xyz_j)
            if max_d_d and d > max_d_d:
                continue
            result.append((rsd_num_j, xyz_j, d))
        if result:
            neighbors_map[rsd_num_i] = result
    return neighbors_map


def neighborhood_rmsd(
    target: daimyo.core.pose.Pose,
    mobile: daimyo.core.pose.Pose,
    max_seq_d: int = 12,
    min_seq_d: int = 4,
    max_d_d: float = 0,
) -> typing.Dict[int, float]:
    """
        RMSD from atom pair distances in neighborhood
        NOTE:
        mobile is a bad name we don't actually move at all
        not_found_ret is what to do if we don't have
    """
    all_target_CAs = get_CA_coords(target)
    all_mobile_CAs = get_CA_coords(mobile)

    all_target_neighbors = get_neighbors_from_CAs(all_target_CAs, max_seq_d, min_seq_d, max_d_d)

    results = {}
    for rsd_num, xyz in all_mobile_CAs.items():
        if rsd_num not in all_target_neighbors:
            continue
        N = 0
        total = 0
        for neighbor in all_target_neighbors[rsd_num]:
            # (neighbor rsd, neighbor xyz, neighbor distance)
            if neighbor[0] not in all_mobile_CAs:
                continue
            d = np.linalg.norm(xyz - all_mobile_CAs[neighbor[0]])
            err = d - neighbor[-1]
            total += err * err
            N += 1
        if N != 0:
            results[rsd_num] = math.sqrt(total / N)
    return results


def aligned_rmsd(target: daimyo.core.pose.Pose, mobile: daimyo.core.pose.Pose) -> float:
    """get aligned rmsd of two pose objects

    Note:
        returns 100 if no aligned residues
    """
    all_target_CAs = get_CA_coords(target)
    all_mobile_CAs = get_CA_coords(mobile)
    return aligned_rmsd_from_CAs(all_target_CAs, all_mobile_CAs)


def calculate_CA_gdt(target_CAs, mobile_CAs):
    """Calculate CA gdt

    assume indicies match!
    """
    distance_cutoffs = {1: 0, 2: 0, 4: 0, 8: 0}
    for i, atom in enumerate(target_CAs):
        d = np.linalg.norm(atom - mobile_CAs[i])
        for k in distance_cutoffs.keys():
            if d <= k:
                distance_cutoffs[k] += 1
    avg = 0
    for key in distance_cutoffs.keys():
        avg += distance_cutoffs[key] / len(target_CAs)
    avg /= 4
    return avg


def aligned_gdt(target: daimyo.core.pose.Pose, mobile: daimyo.core.pose.Pose) -> float:
    all_target_CAs = get_CA_coords(target)
    all_mobile_CAs = get_CA_coords(mobile)

    target_CAs, mobile_CAs = trim_target_and_mobile(all_target_CAs, all_mobile_CAs)

    target_CAs = recenter_coords_at_point(target_CAs, target_CAs.mean(axis=0))
    mobile_CAs = recenter_coords_at_point(mobile_CAs, mobile_CAs.mean(axis=0))

    rotation_matrix, _ = matrix_from_kabsch_align(target_CAs, mobile_CAs)
    mobile_CAs = apply_rot_matrix_to_CA_coords(mobile_CAs, rotation_matrix)
    gdt_ts = calculate_CA_gdt(target_CAs, mobile_CAs)
    return gdt_ts


def gdt(target: daimyo.core.pose.Pose, mobile: daimyo.core.pose.Pose) -> float:
    all_target_CAs = get_CA_coords(target)
    all_mobile_CAs = get_CA_coords(mobile)

    target_CAs, mobile_CAs = trim_target_and_mobile(all_target_CAs, all_mobile_CAs)

    gdt_ts = calculate_CA_gdt(target_CAs, mobile_CAs)
    return gdt_ts


def calculate_CA_rms(target_CAs, mobile_CAs) -> float:
    sum_of_squared_deviations = np.sum((target_CAs - mobile_CAs) ** 2)
    rms = np.sqrt(sum_of_squared_deviations / float(len(target_CAs)))
    return rms


def rms(target: daimyo.core.pose.Pose, mobile: daimyo.core.pose.Pose) -> float:
    all_target_CAs = get_CA_coords(target)
    all_mobile_CAs = get_CA_coords(mobile)

    target_CAs, mobile_CAs = trim_target_and_mobile(all_target_CAs, all_mobile_CAs)

    rms = calculate_CA_rms(target_CAs, mobile_CAs)
    return rms


def align_two_CA_coords(
    all_target_CAs: typing.Dict[int, np.ndarray], all_mobile_CAs: typing.Dict[int, np.ndarray]
) -> typing.Dict[int, np.ndarray]:
    target_CAs, mobile_CAs = trim_target_and_mobile(all_target_CAs, all_mobile_CAs)
    original_target_COM = target_CAs.mean(axis=0)

    target_CAs = recenter_coords_at_point(target_CAs, target_CAs.mean(axis=0))
    mobile_CAs = recenter_coords_at_point(mobile_CAs, mobile_CAs.mean(axis=0))

    rot_matrix, _ = matrix_from_kabsch_align(target_CAs, mobile_CAs)
    mobile_CAs = apply_rot_matrix_to_CA_coords(mobile_CAs, rot_matrix)
    mobile_CAs = recenter_coords_at_point(mobile_CAs, -original_target_COM)
    for i, k in enumerate(all_mobile_CAs.keys()):
        all_mobile_CAs[k] = mobile_CAs[i]
    return all_mobile_CAs


def align_two_poses(
    target: daimyo.core.pose.Pose, mobile: daimyo.core.pose.Pose
) -> (np.ndarray, float, np.ndarray, daimyo.core.pose.Pose):
    all_target_CAs = get_chained_CA_coords(target)
    all_mobile_CAs = get_chained_CA_coords(mobile)
    target_CAs, mobile_CAs = trim_target_and_mobile(all_target_CAs, all_mobile_CAs)
    if not len(target_CAs) or not len(mobile_CAs):
        return None

    # Center to origin and determine rot matrix
    target_COM_delta = target_CAs.mean(axis=0)
    target_CAs = recenter_coords_at_point(target_CAs, target_COM_delta)
    mobile_COM_delta = mobile_CAs.mean(axis=0)
    mobile_CAs = recenter_coords_at_point(mobile_CAs, mobile_COM_delta)
    rot_matrix, rmsd = matrix_from_kabsch_align(target_CAs, mobile_CAs)

    # Apply matrix to mobile pose
    mobile_centered = center_pose(mobile)
    mobile_aligned = apply_rot_matrix_to_pose(mobile_centered, rot_matrix)

    # Move rotated pose to align with target
    all_target_CAs = get_chained_CA_coords(target)
    all_mobile_aligned_CAs = get_chained_CA_coords(mobile_aligned)
    target_CAs, mobile_aligned_CAs = trim_target_and_mobile(all_target_CAs, all_mobile_aligned_CAs)
    mobile_aligned = center_pose(mobile_aligned, -(target_CAs.mean(axis=0) - mobile_aligned_CAs.mean(axis=0)))

    return rot_matrix, rmsd, -mobile_COM_delta + target_COM_delta, mobile_aligned


# ##### POSE STATS #####


def get_COM_of_pose(pose: daimyo.core.pose.Pose) -> np.ndarray:
    com = np.array((0.0, 0.0, 0.0))
    n = 0
    for chain in pose.chains:
        for group in chain.groups:
            for atom in group.atoms:
                com += atom.xyz
                n += 1
    com /= n
    return com


def get_number_of_residues(pose: daimyo.core.pose.Pose) -> int:
    n = 0
    for chain in pose.chains:
        for group in chain.groups:
            n += 1
    return n


# ##### POSE MANIPULATION #######


def center_pose(pose: daimyo.core.pose.Pose, COM: np.ndarray = None) -> daimyo.core.pose.Pose:
    """Center a pose on a point, if not given, center at (0, 0, 0)
    """
    if type(COM) != np.ndarray:
        COM = get_COM_of_pose(pose)
    for chain in pose.chains:
        for group in chain.groups:
            for atom in group.atoms:
                atom.xyz -= COM
    return pose


def apply_rot_matrix_to_pose(pose: daimyo.core.pose.Pose, rot_matrix: np.ndarray) -> daimyo.core.pose.Pose:
    for chain in pose.chains:
        for group in chain.groups:
            for atom in group.atoms:
                atom.xyz = np.dot(atom.xyz, rot_matrix)
    return pose
