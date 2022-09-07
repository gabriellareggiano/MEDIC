#!/usr/bin/env python
""" wrapper to submit relax
"""
import os
import subprocess
import typing
import shlex
from shutil import copyfile
import attr
import pandas as pd
import runners.slurm as slurm
import py_rosetta.rosetta_wrappers.symmetry as symm
import pathlib
from time import sleep

from dask_jobqueue import SLURMCluster
from dask.distributed import Client


def _run(cmd: str, expected_out: str) -> str:
    if os.path.isfile(expected_out):
        print('job finished, returning model')
        return expected_out
    ret = subprocess.run(shlex.split(cmd), 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.STDOUT)
    if ret.returncode:
        with open(f"relax_brk", 'a') as log:
            log.write(f"Failure with command: {cmd}")
        raise RuntimeError(f"Failure with command: {cmd}\n")
    if not os.path.isfile(expected_out):
        raise RuntimeError(f"expected file: {expected_out} not made from command: {cmd}\n")
    return expected_out


@attr.s(auto_attribs=True, repr=False)
class Relax:
    """ wrapper for rosetta fast relax into density
        TODO - maybe dens shouldn't be hardcoded here
        TODO - compatible w/ silent files
        :param str pdb: path to pdb
        :param str mapfile: path to map
        :param float mapreso: resolution of map
        :param int jobs: number of jobs to submit
        :param bool out_silent: output silent files? will use suffix if given
        :param bool dualspace: use dualspace? if not, cartesian
        :param bool bfactor: do bfactor fitting?
        :param float dens_wt: weight on density
        :param int n_out: number structures per job
        :param bool local_relax: perform local relax? good for large structures
        :param int K: size of neighborhood, smaller k = larger neighborhood
        :param int nexp: expands neighborhoods with cycles
        :param str symm_file: rosetta symmetry file 
        :param str out_dir: name of output directory
        :param str out_suffix: suffix to add to relaxed models
    """
    pdb: str
    mapfile: str
    mapreso: float
    jobs: int = 20
    out_silent: bool = True
    dualspace: bool = True
    bfactor: bool = False
    dens_wt: float = 25.0
    n_out: int = 1
    sugars: bool = False
    local_relax: bool = False
    cart_min: bool = False
    K: int = 10
    nexp: int = 3
    symm_file: str = ""
    csts_fname: str = ""
    cst_wt: float = 0
    in_dir: str = "inputs/"
    out_dir: str = "output/"
    out_suffix: str = ""
    flag_fname: str = "relax.flags"
    xml_fname: str = "relax.xml"
    top_pdbs_dir: str = "top_results/"
    walltime: str = "5:00:00"

    def __attrs_post_init__(self):
        if self.out_suffix and self.out_suffix[0] != "_":
            self.out_suffix = "_" + self.out_suffix
        if self.out_dir and self.out_dir[-1] != "/":
            self.out_dir += "/"
        if self.in_dir and self.in_dir[-1] != "/":
            self.in_dir += "/"
    
    def get_flag_file_str(self) -> str:
        # TODO - write to file here??
        """ write flags into str based on inputs
            out_dir and out_suffix not included here, added in at command
            :returns str rosetta flags
        """
        crys_flag = "-cryst::crystal_refine\n" if self.bfactor else ""
        map_flag = f"-edensity::mapfile {self.mapfile}\n" if self.mapfile else ""
        res_flag = f"-edensity::mapreso {self.mapreso:.2f}\n" if self.mapreso else ""
        flag_file_str = (
            "#i/o\n"
            f"-in::file::s {self.pdb}\n"
            f"{map_flag}"
            f"{res_flag}"
            f"-nstruct {self.n_out}\n"
            f"-parser::protocol {self.xml_fname}\n"
            "-ignore_unrecognized_res\n"
            "-missing_density_to_jump\n"
            f"{crys_flag}"
            "#relax options\n"
            "-beta\n"
            "-beta_cart\n"
            "-default_max_cycles 200")
        if self.sugars:
            flag_file_str += ("\n-include_sugars\n"
                              "-alternate_3_letter_codes pdb_sugar\n"
                              "-auto_detect_glycan_connections\n")
        return flag_file_str
        
    def get_xml_file_str(self) -> str:
        # TODO - write to file here??
        """ write xml str based on job inputs 
            :return str xml setup
        """
        use_symm = 1 if self.symm_file else 0
        use_ds = 1 if self.dualspace else 0
        use_c = 1 if not self.dualspace else 0
        sfname = "dens"
        if self.local_relax:
            relax_mover = f"        <LocalRelax name=\"relax\" scorefxn=\"{sfname}\" max_iter=\"100\" ncyc=\"1\" K=\"{self.K}\" nexp=\"{self.nexp}\"/>\n"
        else:
            relax_mover = f"        <FastRelax name=\"relax\" scorefxn=\"{sfname}\" repeats=\"5\" dualspace=\"{use_ds}\" cartesian=\"{use_c}\"/>\n"
        # optional mover setup
        # TODO -  maybe a dictionary instead? more readable that way?
        symm_mover = ""
        prot_sym = ""
        bfac_mover = ""
        prot_bfac = ""
        min_mover = ""
        prot_min = ""
        cst_mover = ""
        prot_cst = ""
        sf_cst_wt = ""
        if self.symm_file:
            symm_mover = f"        <SetupForSymmetry name=\"setupsymm\" definition=\"{self.symm_file}\"/>\n"
            prot_sym = "        <Add mover=\"setupsymm\"/>\n"
        if self.bfactor:
            bfac_mover = "        <BfactorFitting name=\"fit_bs\" max_iter=\"50\" wt_adp=\"0.0005\" init=\"1\" exact=\"1\"/>\n"
            prot_bfac = "        <Add mover=\"fit_bs\"/>\n"
        if self.cart_min:
            min_mover = f"        <MinMover name=\"cartmin\" scorefxn=\"{sfname}\" max_iter=\"200\" tolerance=\"0.0001\" cartesian=\"1\" bb=\"1\" chi=\"1\"/>\n"
            prot_min = "        <Add mover=\"cartmin\"/>\n"
        if self.csts_fname:
            cst_mover = f"        <ConstraintSetMover name=\"setcsts\" add_constraints=\"0\" cst_file=\"{self.csts_fname}\"/>\n"
            prot_cst = "        <Add mover=\"setcsts\"/>\n"
            sf_cst_wt = f"            <Reweight scoretype=\"atom_pair_constraint\" weight=\"{self.cst_wt}\"/>\n"
        xml_str = (
            "<ROSETTASCRIPTS>\n"
            "    <SCOREFXNS>\n"
            f"        <ScoreFunction name=\"{sfname}\" weights=\"beta_cart\" symmetric=\"{use_symm}\">\n"
            f"            <Reweight scoretype=\"elec_dens_fast\" weight=\"{self.dens_wt}\"/>\n"
            f"{sf_cst_wt}"
            "        </ScoreFunction>\n"
            "    </SCOREFXNS>\n"
            "    <MOVERS>\n"
            f"{symm_mover}"
            "        <SetupForDensityScoring name=\"setupdens\"/>\n"
            f"{cst_mover}"
            f"{min_mover}"
            f"{relax_mover}"
            f"{bfac_mover}"
            "    </MOVERS>\n"
            "    <PROTOCOLS>\n"
            "        <Add mover=\"setupdens\"/>\n"
            f"{prot_sym}"
            f"{prot_cst}"
            f"{prot_min}"
            "        <Add mover=\"relax\"/>\n"
            f"{prot_bfac}"
            "    </PROTOCOLS>\n"
            f"    <OUTPUT scorefxn=\"{sfname}\"/>\n"
            "</ROSETTASCRIPTS>"
            )
        return xml_str

    def get_cmds(self) -> typing.List[str]:
        """ return list of strs for each cmd
        """
        rosetta_exe = os.path.join(os.getenv("ROSETTA3", "/home/reggiano/bin/Rosetta/main"),
                        "source/bin/rosetta_scripts.default.linuxgccrelease")
        rosetta_db = os.path.join(os.getenv("ROSETTA3", "/home/reggiano/bin/Rosetta/main"),
                        "database")
        cmds = list()
        for i in range(1,self.jobs+1):
            if self.out_silent:
                fname = f"S_{self.out_suffix}_{i:03d}" if self.out_suffix else f"S_{i:03d}"
                fname += ".silent"
                out_str = f" -out:file:silent {os.path.join(self.out_dir,fname)}"
            else:
                pre_str = f" -out::prefix {self.out_dir}" if self.out_dir else ""
                suf_str = f" -out::suffix {self.out_suffix}_{i:03d}"
                out_str = f"{pre_str}{suf_str}"
            cmd = (f"{rosetta_exe} -database {rosetta_db}"
                   f"{out_str}"
                   f" @{self.flag_fname}")
            cmds.append(cmd)
        return cmds
    
    def get_model_fnames(self) -> typing.List[str]:
        """ return a list of names of each model,
            if pdb output used
        """
        models = list()
        base_name = os.path.basename(self.pdb[:-4])
        for i in range(1,self.jobs+1):
            for j in range(1,self.n_out+1):
                model = (f"{os.path.join(self.out_dir, base_name)}"
                         f"{self.out_suffix}_{i:03d}_{j:04d}.pdb")
                models.append(model)
        return models
    
    def get_sc_fnames(self) -> typing.List[str]:
        """ return a list of names of each model
            TODO - compatible w/ silent file?
        """
        sc_files = list()
        base_name = "score"
        for i in range(1, self.jobs+1):
            sc = (f"{os.path.join(self.out_dir, base_name)}"
                  f"{self.out_suffix}_{i:03d}.sc")
            sc_files.append(sc)
        return sc_files

    def compile_results(self, sc_files: typing.List[str], n: int = 1, iden: str = "") -> typing.List[str]:
        """ put top results based on score into 
            seperate directory
            TODO - silent file compatible??
                :param List[str] sc_files: list of paths to sc files
                :param int n: num of models to return
                :return List[str] top_models: list of paths to top models
        """
        # i kind of want to change this so files just get renamed to their final 
        # ranking, but i think it would screw up guide_image_processing
        # so i will wait until later.
        frames = list()
        for sc in sc_files:
            sc_pd = pd.read_csv(sc, header=1, sep='\s+')
            frames.append(sc_pd)
        all_scores = pd.concat(frames)
        if iden: all_scores = all_scores[all_scores['description'].str.contains(iden)]
        models = all_scores.sort_values('total_score').head(n)['description'].tolist()
        if n > 1:
            os.makedirs(self.top_pdbs_dir, exist_ok=True)
            for i,model in enumerate(models):
                model += ".pdb"
                new_model = os.path.join(self.top_pdbs_dir,f"rank_{i+1:03d}.pdb")
                if not os.path.isfile(new_model) and os.path.isfile(model):
                   copyfile(model, new_model)
                models[i] = os.path.abspath(new_model)
        else:
            for i,model in enumerate(models):
                model += ".pdb"
                new_model = os.path.join(self.out_dir, f"rank_{i+1:03d}.pdb")
                os.rename(model, new_model)
                models[i] = os.path.abspath(new_model)
        return models

    def setup_files(self, copy_input=True) -> None:
        pathlib.Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(self.in_dir).mkdir(exist_ok=True, parents=True)

        if copy_input:
            new_pdb_loc = os.path.join(self.in_dir, "input.pdb")
            copyfile(self.pdb, new_pdb_loc)
            self.pdb = new_pdb_loc

        self.xml_fname = os.path.join(self.in_dir, self.xml_fname)
        with open(self.xml_fname, 'w') as f:
            f.write(self.get_xml_file_str())
        
        self.flag_fname = os.path.join(self.in_dir, self.flag_fname)
        with open(self.flag_fname, 'w') as f:
            f.write(self.get_flag_file_str())
        
        sleep(30) # wait for files to exist

    def run_relax(self, client: Client) -> typing.List[str]:
        """ submit relax jobs with dask
        """
        self.setup_files()
        cmds = self.get_cmds()
        expected_models = self.get_model_fnames()

        tasks = []
        for cmd, model in zip(cmds, expected_models):
            tasks.append(client.submit(_run, cmd, model))
        results = client.gather(tasks)

        return results

    def apply(self, client: Client) -> typing.List[str]:
        final_models = self.run_relax(client)
        basename = os.path.basename(self.pdb[:-4])
        sc_files = self.get_sc_fnames()
        num_top = 1 if self.jobs <= 10 else 5
        sleep(30) # wait for files to exist
        top_models = self.compile_results(sc_files, n=num_top, iden=basename)
        return top_models


def commandline_main():
    import sys
    # TODO - this is currrently for testing, NOT for running!
    relax = Relax(sys.argv[1], sys.argv[2], float(sys.argv[3]),
        out_silent=False, local_relax=True, bfactor=True, cart_min=True, jobs=2)
    with SLURMCluster(cores=1,
        memory=f"10GB",
        queue="cpu",
        walltime=relax.walltime,
        job_name="relax",
        env_extra=["source ~/.bashrc"]
    ) as cluster:
        cluster.adapt(minimum=0, maximum=relax.jobs)
        with Client(cluster) as client:
            models = relax.apply(client)


if __name__ == "__main__":
    commandline_main()
