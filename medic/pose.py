#!/usr/bin/env python

import typing
import attr
import numpy as np


@attr.s(auto_attribs=True)
class Atom:
    xyz: typing.Any
    atomName: str
    element: str
    occupancy: float = None
    bfactor: float = None
    charge: int = None

    def distance(self, a2):
        return np.linalg.norm(self.xyz - a2.xyz)


@attr.s(auto_attribs=True)
class Group:
    groupName: str
    groupNumber: int
    singleLetter: str
    chemCompType: str
    atoms: typing.List[Atom] = attr.Factory(list)

    def atom_by_name(self, name: str):
        for atom in self.atoms:
            if atom.atomName == name:
                return atom
        return None

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        return (
            f"Group: groupName: {self.groupName} groupNumber: {self.groupNumber} singleLetter:"
            f" {self.singleLetter} chemCompType: {self.chemCompType} num_atoms: {len(self.atoms)}"
        )


@attr.s(auto_attribs=True)
class Chain:
    """A continuous seq of groups

    This isn't just residues with the same chainID. This
    is meant to be a sequence of continuous residues.
    note that we use ID when we write pdbs
    """

    ID: str
    name: str = None
    groups: typing.List[Group] = attr.Factory(list)

    def __attrs_post_init__(self):
        if not self.name:
            self.name = self.ID

    def __len__(self):
        return len(self.groups)

    def __str__(self):
        return f"Chain: ID: {self.ID} name: {self.name} num groups: {len(self.groups)}"


@attr.s(auto_attribs=True)
class Pose:
    """A pose object

    :param chains: a list of chains that the pose contains
    :type chains: typing.List[Chain]
    :param ID: the ID of this pose
    :type ID: string
    """

    chains: typing.List[Chain] = attr.Factory(list)
    ID: typing.Optional[str] = None

    def __len__(self):
        return len(self.chains)
