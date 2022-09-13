#!/usr/bin/env python
""" submit relax through pyrosetta
"""
import pyrosetta
import typing
import attr
import argparse

@attr.s(auto_attribs=True, repr=False)
class Relax:
    """ wrapper for XML protocol with pyrosetta
    """
    mapfile: str
    mapreso: float
    dens_wt: float = 50.0
    n_out: int = 1
    K: int = 10
    nexp: int = 3
    
    def get_flag_file_str(self) -> str:
        """ write flags into str based on inputs
        """
        map_flag = f"-edensity::mapfile {self.mapfile}\n" 
        res_flag = f"-edensity::mapreso {self.mapreso:.2f}\n" 
        flag_file_str = (
            f"{map_flag}"
            f"{res_flag}"
            "-cryst::crystal_refine\n"
            "-ignore_unrecognized_res\n"
            "-missing_density_to_jump\n"
            "-beta\n"
            "-beta_cart\n"
            "-default_max_cycles 200")
        return flag_file_str
        
    def get_xml_file_str(self) -> str:
        """ write xml str based on job inputs 
            :return str xml setup
        """
        sfname = "dens"
        xml_str = (
            "<ROSETTASCRIPTS>\n"
            "    <SCOREFXNS>\n"
            f"        <ScoreFunction name=\"{sfname}\" weights=\"beta_cart\">\n"
            f"            <Reweight scoretype=\"elec_dens_fast\" weight=\"{self.dens_wt}\"/>\n"
            "        </ScoreFunction>\n"
            "    </SCOREFXNS>\n"
            "    <MOVERS>\n"
            "        <SetupForDensityScoring name=\"setupdens\"/>\n"
            f"        <MinMover name=\"cartmin\" scorefxn=\"{sfname}\" max_iter=\"200\" tolerance=\"0.0001\" cartesian=\"1\" bb=\"1\" chi=\"1\"/>\n"
            f"        <LocalRelax name=\"relax\" scorefxn=\"{sfname}\" max_iter=\"100\" ncyc=\"1\" K=\"{self.K}\" nexp=\"{self.nexp}\"/>\n"
            "        <BfactorFitting name=\"fit_bs\" max_iter=\"50\" wt_adp=\"0.0005\" init=\"1\" exact=\"1\"/>\n"
            "    </MOVERS>\n"
            "    <PROTOCOLS>\n"
            "        <Add mover=\"setupdens\"/>\n"
            "        <Add mover=\"cartmin\"/>\n"
            "        <Add mover=\"relax\"/>\n"
            "        <Add mover=\"fit_bs\"/>\n"
            "    </PROTOCOLS>\n"
            f"    <OUTPUT scorefxn=\"{sfname}\"/>\n"
            "</ROSETTASCRIPTS>"
            )
        return xml_str

    def apply(self, pose: pyrosetta.rosetta.core.pose.Pose) -> None:
        """ get xml string and apply to given pose
        """
        xml_str = self.get_xml_file_str()
        xml_obj = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml_str)
        xml_prot = xml_obj.get_mover("ParsedProtocol")
        xml_prot.apply(pose)


def commandline_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True)
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--reso', type=float, required=True)
    args = parser.parse_args()

    out_pdb = f"{args.pdb[:-4]}_0001.pdb"
    relax = Relax(args.map, args.reso)
    flag_str = relax.get_flag_file_str()

    pyrosetta.init(flag_str)
    pose = pyrosetta.pose_from_file(args.pdb)
    relax.apply(pose)
    pose.dump_pdb(out_pdb)


if __name__ == "__main__":
    commandline_main()
