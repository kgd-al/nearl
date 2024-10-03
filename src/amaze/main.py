import logging
import math
import pprint
import random
from pathlib import Path
from typing import Callable, Any

from amaze import MazeWidget, Simulation, Maze, Robot

from abrain import Genome
from abrain.neat.evolver import NEATEvolver as Evolver, NEATConfig as Config
from brain import Brain
from utils import merge_trajectories

# str_mazes = ["M9_4x4_U"]
str_mazes = ["M9_4x4_U", "M5_4x4_U", "M7_4x4_U", "M0_4x4_U"]
mazes = [m
         for s in str_mazes
         for m in Maze.from_string(s).all_rotations()[0:1]]
print(mazes)
robot = Robot.BuildData.from_string("H5")
print(robot)


def _evaluate(genome: Genome, traj: bool, fn: Callable[[Simulation], Any]):
    ann = Brain(genome, robot)
    for maze in mazes:
        ann.reset()
        simulation = Simulation(maze, robot, save_trajectory=traj)
        simulation.run(ann)
        yield fn(simulation)


optimal_fitness = 1 #len(maze.solution)


def evaluate(genome: Genome):
    values = list(_evaluate(genome, False, lambda s: s.normalized_reward()))
    return sum(values) / len(mazes)


seed = 0
pop, gen = 100, 100#100
species = 8
path = Path("tmp/amaze")

gen_digits = math.ceil(math.log10(gen+1))

logging.basicConfig(level=logging.INFO)

rng = random.Random(seed)
config = Config(
    seed=seed,
    threads=4,
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

evolver = Evolver(config,
                  evaluator=evaluate,
                  genome_class=Genome,
                  genome_data=dict(data=Genome.Data.create_for_eshn_cppn(
                      dimension=3, seed=seed,
                      with_input_bias=True, with_input_length=True,
                      with_leo=True, with_output_bias=True,
                      with_innovations=True, with_lineage=True
                  )))


def trajectory():
    merge_trajectories(
        list(_evaluate(evolver.champion.genome, True,
             lambda s: MazeWidget.plot_trajectory(s, 500))),
        path.joinpath(f"trajectory{evolver.generation:0{gen_digits}d}.png"))


print(f"{optimal_fitness=}")

with evolver:
    trajectory()
    for g in range(gen):
        evolver.step()
        trajectory()
        if evolver.best_fitness >= optimal_fitness:
            logging.warning("Optimal fitness reached")
            break

# evolver.run(gen)
evolver.generate_plots("pdf",
                       options=dict(optimal_fitness=optimal_fitness))
