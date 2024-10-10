import argparse
import json
from pathlib import Path

from abrain import Genome

parser = argparse.ArgumentParser(
    description="NeuroEvolution and Reinforcement Learning testbed"
                " (evaluation)")
parser.add_argument("--genome", dest="genomes", type=Path,
                    help="Agent to evaluate", action='append', required=True)
parser.add_argument("maze", nargs="+",
                    help="(additional) maze to evaluate the agent on")
parser.add_argument("--rotations", default="all",
                    help="Only only those specific maze rotations"
                         " (instead of all)")
parser.add_argument("--monitor", action="store_true",
                    default=False, help="Generate ann dynamics files")
parser.add_argument("--trajectory", action="store_true",
                    default=False, help="Generate ann dynamics files")
parser.add_argument("--render", nargs="?",
                    default=None, const="png",
                    help="Render genome to file (default png)")
options = parser.parse_args()
print(options)

for genome_path in options.genomes:
    with open(genome_path, "r") as gf:
        genome = Genome.from_json(json.load(gf))
        print(genome)
