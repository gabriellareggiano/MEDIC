#!/usr/bin/env python
""" run refinement in pyrosetta
"""
import pyrosetta
import argparse

def run(pdbf: str, mapf: str,
        reso: float, out_pdb: str) -> None:
    """ setup refinement into density map
        refinement is three steps: cart min, local relax, bfactor fitting
        input pose will be refined
    """
    flags = ['-mute all',
            '-ignore_unrecognized_res',
            '-default_max_cycles 200',
            f'-edensity::mapfile {mapf}',
            f'-edensity::mapreso {reso}']
    pyrosetta.init(' '.join(flags))
    pyrosetta.rosetta.basic.options.set_boolean_option("in:missing_density_to_jump", True)
    pyrosetta.rosetta.basic.options.set_boolean_option("cryst:crystal_refine", True)
    pyrosetta.rosetta.basic.options.set_boolean_option("corrections:shapovalov_lib_fixes_enable", True)
    pyrosetta.rosetta.basic.options.set_file_option("corrections:score:rama_pp_map","scoring/score_functions/rama/fd_beta_nov2016")

    pose = pyrosetta.pose_from_file(pdbf)

    # set up scorefunction
    sf = pyrosetta.create_score_function("ref2015_cart")
    score_manager = pyrosetta.rosetta.core.scoring.ScoreTypeManager()
    dens_scterm = score_manager.score_type_from_name("elec_dens_fast")
    sf.set_weight(dens_scterm, 50.0)

    # set up movers
    movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)

    setupdens_mover = pyrosetta.rosetta.protocols.electron_density.SetupForDensityScoringMover()

    cartmin_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
    cartmin_mover.cartesian(True)
    cartmin_mover.tolerance(1E-4)
    cartmin_mover.set_movemap(movemap)
    cartmin_mover.score_function(sf)

    local_relax_mover = pyrosetta.rosetta.protocols.relax.LocalRelax()
    local_relax_mover.set_K(10)
    local_relax_mover.set_ncyc(1)
    local_relax_mover.set_nexp(3)
    local_relax_mover.set_max_iter(100)
    local_relax_mover.set_sfxn(sf)

    bfacfit_mover = pyrosetta.rosetta.protocols.electron_density.BfactorFittingMover()
    bfacfit_mover.set_max_iter(50)
    bfacfit_mover.set_wt_adp(5E-4)
    bfacfit_mover.set_init(1)
    bfacfit_mover.set_exact(1)

    # apply movers
    setupdens_mover.apply(pose)
    cartmin_mover.apply(pose)
    local_relax_mover.apply(pose)
    bfacfit_mover.apply(pose)

    # score pose with sf so etable is updated
    score = sf(pose)
    pose.dump_pdb(out_pdb)


def commandline_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True)
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--reso', type=float, required=True)
    args = parser.parse_args()

    out_pdb = f"{args.pdb[:-4]}_0001.pdb"
    run(args.pdb, args.map, args.reso, out_pdb)


if __name__ == "__main__":
    commandline_main()
