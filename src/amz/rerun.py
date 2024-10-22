import argparse
import pprint
import time
from datetime import timedelta
from pathlib import Path
from textwrap import indent
from typing import List

import humanize
import numpy as np
from amaze import Maze, Robot, StartLocation, Simulation, MazeWidget

from abrain.core.genome import Genome
from brain import create_genome_data, Brain, controller_data, load, save
from utils import merge_trajectories

genome_data = create_genome_data(seed=0)


def pretty_print(something):
    print(indent(pprint.pformat(something), "  "))


def process(path: Path, mazes: List[Maze], options: argparse.Namespace):
    folder = options.folder or path.parent
    base = folder.joinpath(path.stem)

    if len(mazes) == 0:
        return base, None

    if path.suffix == ".dot":
        dot_path = path
        path = base.with_suffix(".dna")
        save(Genome.from_dot(genome_data, dot_path),
             Robot.BuildData.from_string("H7"),
             path)

    genome, robot = load(path)

    if options.render is not None:
        genome_dot = base.with_suffix(options.render)
        genome_dot = genome.to_dot(genome_data, genome_dot)
        print(path, "rendered to", genome_dot)

    if options.math is not None:
        genome_math = genome.to_math(genome_data, base, extension=options.math)
        print(path, "equation system written to", genome_math)

    simulate = options.monitor or options.trajectory
    trajectories = []

    rewards = None
    if simulate:
        base.mkdir(exist_ok=True, parents=True)

        controller = Brain(genome, robot, labels=True)
        data = controller_data(controller)

        if options.render_3D:
            controller.ann.render3D(controller.labels).write_html(base.joinpath("ann.html"))

        rewards = []
        for maze in mazes:
            folder = base.joinpath(maze.to_string())
            folder.mkdir(exist_ok=True, parents=False)

            controller.reset()

            if options.monitor:
                controller.monitor(folder)

            simulation = Simulation(maze, robot, save_trajectory=options.trajectory, deadline_factor=1)
            simulation.run(controller)
            rewards.append(simulation.normalized_reward())

            if options.monitor:
                controller.stop_monitor()

            if options.trajectory:
                trajectories.append(MazeWidget.plot_trajectory(simulation, 500))
                trajectories[-1].save(str(folder.joinpath("trajectory.png")))

        if options.trajectory and options.merge_trajectories:
            merge_trajectories(trajectories, base.with_suffix(".trajectories.png"))

        print("Rewards:", rewards)
        print("  range:", np.quantile(rewards, [0, .5, 1]))
        print("    avg:", np.average(rewards), "+-", np.std(rewards))

    return base, rewards


def main():
    parser = argparse.ArgumentParser(
        description="NeuroEvolution and Reinforcement Learning testbed"
                    " (evaluation)")
    parser.add_argument("-v", dest="verbose", action="count", default=0)
    parser.add_argument("-q", dest="quiet", action="count", default=0)
    parser.add_argument("--genome", dest="genomes", type=Path,
                        help="Agent to evaluate", action='append', required=True)
    parser.add_argument("maze", nargs="+",
                        help="(additional) maze to evaluate the agent on")
    parser.add_argument("--rotations", default="all",
                        help="Only only those specific maze rotations"
                             " (instead of all)")
    parser.add_argument("--monitor", action="store_true",
                        default=False, help="Generate ann dynamics files")
    parser.add_argument("--trajectory", const=1, nargs='?',
                        default=0, help="Render the agent's trajectory to file."
                                        " Provided value controls verbosity.")
    parser.add_argument("--render", nargs="?",
                        default=None, const="png",
                        help="Render genome to file (default png)")
    parser.add_argument("--render-3D", action='store_true',
                        default=False, help="Decode and render the ANN to file")
    parser.add_argument("--math", nargs="?",
                        default=None, const="math.png",
                        help="Generate equation system for genome to file (default math.png)")
    parser.add_argument("--folder", type=Path, default=None,
                        help="Where to put the output files (defaults to the"
                             "provided genomes' parent directory)")
    options = parser.parse_args()

    verbosity = options.verbose - options.quiet
    if verbosity >= 0:
        print("Provided options:")
        pretty_print(options.__dict__)

    if options.render[0] != ".":
        options.render = "." + options.render

    if options.folder is not None:
        options.folder.mkdir(exist_ok=True, parents=True)

    mazes = options.maze
    if options.rotations == "all":
        mazes = [_m for m in mazes for _m in Maze.from_string(m).all_rotations()]
    else:
        rotations = options.rotations.split(",")
        mazes = [
            Maze.generate(
                Maze.BuildData.from_string(m).where(start=StartLocation.from_shorthand(r)))
            for m in mazes for r in rotations
        ]

    if verbosity >= 0:
        print("Using mazes:")
        pretty_print(mazes)

    start_time = time.perf_counter()

    for genome_path in options.genomes:
        process(genome_path, mazes, options)

    print("Evaluated", len(options.genomes), "agent(s) on", len(mazes), "maze(s) in",
          humanize.precisedelta(timedelta(seconds=time.perf_counter() - start_time)))


if __name__ == "__main__":
    main()