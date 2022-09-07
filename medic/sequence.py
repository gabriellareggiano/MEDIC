"""
Sequence / alignment related stuff
"""

from dataclasses import dataclass
from typing import Optional

import attr


@dataclass
class Sequence:
    """
    Class to store fasta information
    """

    name: str
    sequence: str
    parent: Optional["Sequence"] = None

    def to_fasta(self):
        s = f">{self.name}{ ' parent: ' + self.parent.name if self.parent else ''}\n{self.sequence}"
        return s

    def __repr__(self):
        return self.to_fasta()

    def to_dict(self):
        return attr.asdict(self)
