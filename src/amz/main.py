import argparse
import json
import logging
import math
import os
import pprint
import random
import time
import humanize
from datetime import timedelta
from logging import debug
from pathlib import Path
from typing import Callable, Any, Optional

from amaze import MazeWidget, Simulation, Maze, Robot

from abrain import Genome
from abrain.neat.evolver import NEATEvolver as Evolver, NEATConfig as Config
from brain import Brain, create_genome_data, controller_data, save
from utils import merge_trajectories


parser = argparse.ArgumentParser(
    description="NeuroEvolution and Reinforcement Learning testbed"
                " (evolution)")
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--genome-seed", default=None, type=Path)
options = parser.parse_args()


def all_mazes(seeds: list, size: int, mclass: str):
    suffix = f"_{size}x{size}_{mclass}"
    return [m
            for s in seeds
            for m in Maze.from_string(f"M{s}{suffix}").all_rotations()]


trivial_mazes = all_mazes([9, 5, 7, 0], 4, "U")
simple_mazes = all_mazes([15, 1, 3, 12], 5, "C1")

mazes = simple_mazes
print(mazes)
robot = Robot.BuildData.from_string("H7")
print(robot)


def _evaluate(genome: Genome, fn: Callable[[dict, Simulation], Any],
              details: int, _path: Optional[Path] = None):

    controller = Brain(genome, robot, labels=False)
    data = controller_data(controller)

    if details:
        _path.mkdir(exist_ok=True, parents=True)
        save(genome, robot, _path.joinpath("genome.dna"))

    for maze in mazes:
        controller.reset()
        simulation = Simulation(maze, robot, save_trajectory=details)
        simulation.run(controller)
        yield fn(data, simulation)


optimal_fitness = 1 #len(maze.solution)


def evaluate(genome: Genome):
    def fitness(data: dict, simulation: Simulation):
        return simulation.normalized_reward() + data["fitness_offset"]

    values = list(_evaluate(genome, fitness, details=False))
    return sum(values) / len(mazes)


seed = options.seed
if seed is None:
    seed = time.time_ns()

pop, gen = 100, 100#100
species = 8
path = Path(f"tmp/amaze/{seed}")

gen_digits = math.ceil(math.log10(gen+1))

logging.basicConfig(level=logging.INFO)

rng = random.Random(seed)
config = Config(
    seed=seed,
    threads=os.cpu_count()-1,
    log_dir=path,
    log_level=4,
    population_size=pop,
    species_count=species,
    elitism=2,
    # initial_distance_threshold=5,
    initial_distance_threshold=10,
    overwrite=True,
)
pprint.pprint(config)

genome_data = create_genome_data(seed)

if options.genome_seed:
    genome_seed = Genome.from_dot(genome_data, options.genome_seed)
    random_genome = lambda *_, data: genome_seed
else:
    random_genome = Genome.random

evolver = Evolver(config,
                  evaluator=evaluate,
                  genome_class=Genome,
                  genome_data=dict(data=genome_data),
                  random=random_genome)

if options.genome_seed:
    genome_seed.to_dot(genome_data, path.joinpath("genome_seed.png"))

trajectories_path = path.joinpath("trajectory")
trajectories_path.mkdir(exist_ok=True, parents=True)


def trajectory(_improved: bool, _final: bool):
    _id = f"{evolver.generation:0{gen_digits}d}"
    merge_trajectories(
        list(_evaluate(evolver.champion.genome,
                       lambda _, s: MazeWidget.plot_trajectory(s, 500),
                       details=int(_improved) + 2 * int(_final),
                       _path=path.joinpath(_id))),
        trajectories_path.joinpath(f"{_id}.png"))


print(f"{optimal_fitness=}")

with evolver:
    start_time = time.perf_counter()

    trajectory(_improved=True, _final=False)
    for g in range(gen):
        prev_best = evolver.best_fitness
        evolver.step()

        should_break = (evolver.best_fitness >= optimal_fitness)
        improved = (evolver.best_fitness > prev_best)
        end = should_break or (g == gen-1)
        if improved or end:
            trajectory(_improved=improved, _final=end)

        if should_break:
            logging.warning("Optimal fitness reached")
            break

    print("Evolution finished in", humanize.precisedelta(timedelta(seconds=time.perf_counter() - start_time)))

evolver.generate_plots("png",
                       options=dict(optimal_fitness=optimal_fitness))
