"""
Aligning sequences:
You must NOT have '*' in your sequence, if you do, things will break!
"""

from typing import Union, Tuple, List
from enum import IntEnum
from dataclasses import dataclass

import attr
import numpy as np


@dataclass(repr=False)
class Alignment(object):
    """
    Class to store an alignment as a part of an hhpred result
    """

    target_start: int
    target_seq: str
    template_start: int
    template_seq: str
    template_dssp: str

    def __str__(self):
        return (
            "target_seq/template_seq\n"
            + f"{self.target_start} {self.target_seq}\n"
            + f"{self.template_start} {self.template_seq}\n"
            + f"{self.template_dssp}"
        )


class AlignMove(IntEnum):
    diagonal = 1
    left = 2
    above = 3
    end = 4


class Cell:
    __slots__ = ("score_", "type_", "coords_", "next_")

    def __init__(
        self,
        score: float,
        type_n: AlignMove,
        coord: np.ndarray,
        next_n: Union["Cell", None],
    ):
        self.score_ = score
        self.type_ = type_n
        self.coords_ = coord
        self.next_ = next_n


@attr.s(auto_attribs=True)
class SmithWaterman:
    match_score: float = 1
    mismatch_score: float = -100
    gap_open_penalty: float = -1
    gap_extension_penalty: float = -1
    threshold: float = 0

    def build_matrix(self, new_seq_1, new_seq_2):
        new_seq_1_len = len(new_seq_1)
        new_seq_2_len = len(new_seq_2)

        best_cell = Cell(0, AlignMove.end, np.array((0, 0)), None)

        scoring_matrix = [
            [
                Cell(0.0, AlignMove.end, np.array((i, j)), None)
                for j in range(new_seq_2_len)
            ]
            for i in range(new_seq_1_len)
        ]

        for y in range(1, new_seq_2_len):
            for x in range(1, new_seq_1_len):
                left_gap_penalty = self.gap_open_penalty
                up_gap_penalty = self.gap_open_penalty
                if scoring_matrix[x][y - 1].type_ == AlignMove.above:
                    up_gap_penalty = self.gap_extension_penalty
                if scoring_matrix[x - 1][y].type_ == AlignMove.left:
                    left_gap_penalty = self.gap_extension_penalty

                up_gap = scoring_matrix[x][y - 1].score_ + up_gap_penalty
                left_gap = scoring_matrix[x - 1][y].score_ + left_gap_penalty
                this_score = (
                    self.match_score
                    if new_seq_1[x] == new_seq_2[y]
                    else self.mismatch_score
                )
                mm = scoring_matrix[x - 1][y - 1].score_ + this_score

                this_cell = scoring_matrix[x][y]
                if mm > left_gap and mm > up_gap and mm >= self.threshold:
                    this_cell.score_ = mm
                    this_cell.next_ = scoring_matrix[x - 1][y - 1]
                    this_cell.type_ = AlignMove.diagonal
                elif (
                    left_gap >= mm and left_gap >= up_gap and left_gap >= self.threshold
                ):
                    this_cell.score_ = left_gap
                    this_cell.next_ = scoring_matrix[x - 1][y]
                    this_cell.type_ = AlignMove.left
                elif up_gap >= mm and up_gap >= left_gap and up_gap >= self.threshold:
                    this_cell.score_ = up_gap
                    this_cell.next_ = scoring_matrix[x][y - 1]
                    this_cell.type_ = AlignMove.above
                else:
                    this_cell.score_ = self.threshold
                    this_cell.type_ = AlignMove.end

                if this_cell.score_ > best_cell.score_:
                    best_cell = this_cell
        return scoring_matrix, best_cell

    def traceback_matrix(
        self,
        new_seq_1: str,
        new_seq_2: str,
        scoring_matrix: List[List[Cell]],
        best_cell: Cell,
    ):
        """
        We use '*' for gaps because we want to be able to align to sequences with '-' in them!
        """
        current_cell = best_cell

        aligned_seq_1 = ""
        aligned_seq_2 = ""

        while True:
            x = current_cell.coords_[0]
            y = current_cell.coords_[1]

            if current_cell.type_ == AlignMove.diagonal:
                aligned_seq_1 = new_seq_1[x] + aligned_seq_1
                aligned_seq_2 = new_seq_2[y] + aligned_seq_2
            elif current_cell.type_ == AlignMove.left:
                aligned_seq_1 = new_seq_1[x] + aligned_seq_1
                aligned_seq_2 = "*" + aligned_seq_2
                new_seq_2 = new_seq_2[: y + 1] + "*" + new_seq_2[y + 1 :]
            elif current_cell.type_ == AlignMove.above:
                aligned_seq_1 = "*" + aligned_seq_1
                new_seq_1 = new_seq_1[: x + 1] + "*" + new_seq_1[x + 1 :]
                aligned_seq_2 = new_seq_2[y] + aligned_seq_2
            else:
                error_text = (
                    f"{aligned_seq_1}\n"
                    f"{aligned_seq_2}\n"
                    f"{current_cell}\n"
                    "seq_1-2\n"
                    f"{new_seq_1}\n-\n"
                    f"{new_seq_2}\n---"
                )
                raise RuntimeError(
                    f"Found a pointer that doesn't go anywere!\n{error_text}"
                )

            if not current_cell.next_.next_:
                break
            current_cell = current_cell.next_

        return aligned_seq_1, aligned_seq_2, current_cell

    def build_gapped_sequence(
        self,
        new_seq_1: str,
        new_seq_2: str,
        aligned_seq_1: str,
        aligned_seq_2: str,
        final_cell: Cell,
    ) -> Tuple[str, str]:
        """
        """

        max_pre = max([final_cell.coords_[0], final_cell.coords_[1]])
        gapless_1 = "".join([x for x in aligned_seq_1 if x != "*"])

        gapless_2 = [x for x in aligned_seq_2 if x != "*"]

        aligned_seq_1 = (
            "-" * (max_pre - (final_cell.coords_[0]))
            + new_seq_1[1 : final_cell.coords_[0]]
            + aligned_seq_1
            + new_seq_1[len(gapless_1) + final_cell.coords_[0] :]
        )
        aligned_seq_2 = (
            "-" * (max_pre - (final_cell.coords_[1]))
            + new_seq_2[1 : final_cell.coords_[1]]
            + aligned_seq_2
            + new_seq_2[len(gapless_2) + final_cell.coords_[1] :]
        )

        total_max = max([len(aligned_seq_1), len(aligned_seq_2)])

        aligned_seq_1 += "-" * (total_max - len(aligned_seq_1))
        aligned_seq_2 += "-" * (total_max - len(aligned_seq_2))

        aligned_seq_1 = aligned_seq_1.replace("*", "-")
        aligned_seq_2 = aligned_seq_2.replace("*", "-")

        return aligned_seq_1, aligned_seq_2

    def apply(self, seq_1: str, seq_2: str) -> Tuple[Tuple[str, int], Tuple[str, int]]:
        new_seq_1 = f"-{seq_1}"
        new_seq_2 = f"-{seq_2}"
        scoring_matrix, best_cell = self.build_matrix(new_seq_1, new_seq_2)
        aligned_seq_1, aligned_seq_2, final_cell = self.traceback_matrix(
            new_seq_1, new_seq_2, scoring_matrix, best_cell
        )
        aligned_seq_1, aligned_seq_2 = self.build_gapped_sequence(
            new_seq_1, new_seq_2, aligned_seq_1, aligned_seq_2, final_cell
        )
        return aligned_seq_1, aligned_seq_2


@attr.s(auto_attribs=True)
class NeedlemanWunsch:
    gap_open_penalty: float = -1
    gap_extension_penalty: float = -1
    match_score: int = 1
    mismatch_score: int = -2

    def traceback_matrix(self, new_seq_1, new_seq_2, scoring_matrix, best_cell):
        current_cell = best_cell

        aligned_seq_1 = ""
        aligned_seq_2 = ""

        count = 0
        while True:
            count += 1
            x = current_cell.coords_[0]
            y = current_cell.coords_[1]
            # print(best_cell.coords_, best_cell.score_)
            # print(aligned_seq_1)
            # print(aligned_seq_2)

            if current_cell.type_ == AlignMove.diagonal:
                aligned_seq_1 = new_seq_1[x] + aligned_seq_1
                aligned_seq_2 = new_seq_2[y] + aligned_seq_2
            elif current_cell.type_ == AlignMove.left:
                aligned_seq_1 = new_seq_1[x] + aligned_seq_1
                aligned_seq_2 = "-" + aligned_seq_2
                new_seq_2 = new_seq_2[: y + 1] + "-" + new_seq_2[y + 1 :]
            elif current_cell.type_ == AlignMove.above:
                aligned_seq_1 = "-" + aligned_seq_1
                new_seq_1 = new_seq_1[: x + 1] + "-" + new_seq_1[x + 1 :]
                aligned_seq_2 = new_seq_2[y] + aligned_seq_2
            else:
                print("ERROR!")
                print(aligned_seq_1)
                print(aligned_seq_2)
                raise RuntimeError("Found a pointer that doesn't go anywere!")

            if current_cell.next_.type_ == AlignMove.end:
                break
            current_cell = current_cell.next_
        return aligned_seq_1, aligned_seq_2, current_cell

    def build_matrix(self, new_seq_1, new_seq_2):
        new_seq_1_len = len(new_seq_1)
        new_seq_2_len = len(new_seq_2)

        # INIT
        scoring_matrix = []
        for i in range(new_seq_1_len):
            scoring_matrix.append(
                [
                    Cell(0.0, AlignMove.end, np.array((i, j)))
                    for j in range(new_seq_2_len)
                ]
            )
        for i in range(new_seq_2_len):
            if i != 0:
                scoring_matrix[0][i].score_ = self.gap_open_penalty + (
                    (i - 1) * self.gap_extension_penalty
                )
                scoring_matrix[0][i].next_ = scoring_matrix[0][i - 1]
                scoring_matrix[0][i].type_ = AlignMove.left
        for i in range(new_seq_1_len):
            if i != 0:
                scoring_matrix[i][0].score_ = self.gap_open_penalty + (
                    (i - 1) * self.gap_extension_penalty
                )
                scoring_matrix[i][0].next_ = scoring_matrix[i - 1][0]
                scoring_matrix[i][0].type_ = AlignMove.above

        for y in range(1, new_seq_2_len):
            for x in range(1, new_seq_1_len):
                left_gap_penalty = self.gap_open_penalty
                up_gap_penalty = self.gap_open_penalty
                if scoring_matrix[x][y - 1].type_ == AlignMove.above:
                    up_gap_penalty = self.gap_extension_penalty
                if scoring_matrix[x - 1][y].type_ == AlignMove.left:
                    left_gap_penalty = self.gap_extension_penalty
                up_gap = scoring_matrix[x][y - 1].score_ + up_gap_penalty
                left_gap = scoring_matrix[x - 1][y].score_ + left_gap_penalty
                this_score = (
                    self.match_score
                    if new_seq_1[x] == new_seq_2[y]
                    else self.mismatch_score
                )
                mm = scoring_matrix[x - 1][y - 1].score_ + this_score

                this_cell = scoring_matrix[x][y]
                if mm >= left_gap and mm >= up_gap:
                    this_cell.score_ = mm
                    this_cell.next_ = scoring_matrix[x - 1][y - 1]
                    this_cell.type_ = AlignMove.diagonal
                elif left_gap >= mm and left_gap >= up_gap:
                    this_cell.score_ = left_gap
                    this_cell.next_ = scoring_matrix[x - 1][y]
                    this_cell.type_ = AlignMove.left
                else:
                    this_cell.score_ = up_gap
                    this_cell.next_ = scoring_matrix[x][y - 1]
                    this_cell.type_ = AlignMove.above
                if y == new_seq_2_len - 1 and x == new_seq_1_len - 1:
                    # 543                   AlignMove.diagonal   543          -2    I              A
                    print(
                        this_cell.score_,
                        this_cell.type_,
                        mm,
                        this_score,
                        new_seq_1[x],
                        new_seq_2[y],
                    )

        best_cell = scoring_matrix[new_seq_1_len - 1][new_seq_2_len - 1]
        for x in scoring_matrix:
            print(",\t".join([f"{(int(y.score_), y.type_.name)}" for y in x]))
        return self.traceback_matrix(new_seq_1, new_seq_2, scoring_matrix, best_cell)

    def apply(self, seq1, seq2):
        new_seq_1 = f"-{seq1}"
        new_seq_2 = f"-{seq2}"
        aligned_seq_1, aligned_seq_2, final_cell = self.build_matrix(
            new_seq_1, new_seq_2
        )
        return (
            (aligned_seq_1, final_cell.coords_[0]),
            (aligned_seq_2, final_cell.coords_[1]),
        )

        # score_matrix, traceback_matrix = self.matrix_initialization(seq1, seq2)
        # score, score_matrix, traceback_matrix = self.scoring(seq1, seq2, score_matrix, traceback_matrix)
        # aligned_seq1, aligned_seq2 = self.alignment(seq1, seq2, traceback_matrix)
        # kreturn score_matrix, score, aligned_seq1, aligned_seq2
