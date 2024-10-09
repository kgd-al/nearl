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
from brain import Brain
from utils import merge_trajectories


parser = argparse.ArgumentParser(description="NeuroEvolution and Reinforcement Learning testbed")
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--genome-seed", default=None, type=Path)
options = parser.parse_args()

maze_size = "10x10" #"10x10"
# maze_class = "U"
maze_class = "C1"
# str_mazes = ["M9_4x4_U"]
str_mazes = [f"M{ms}_{maze_size}_{maze_class}" for ms in [9, 5, 7, 0]]
mazes = [m
         for s in str_mazes
         for m in Maze.from_string(s).all_rotations()]
print(mazes)
robot = Robot.BuildData.from_string("H5")
print(robot)


def controller_data(controller: Brain):
    ann = controller.ann

    edges = [0, 0, 0, 0]

    for n in ann.neurons():
        if n.type == n.Type.I:
            continue
        for l in n.links():
            _n = l.src()
            if _n.type != n.Type.I:
                continue

            x, _, z = _n.pos.tuple()
            if max(abs(x), abs(z)) < 1:
                continue

            if abs(x) == 1:
                edges[int((x+1)//2)] += 1

            if abs(z) == 1:
                edges[int(2 + (z+1)//2)] += 1

    return dict(
        fitness_offset=(
            -2 if ((sum((len(n.links()) > 0) for n in ann.output_neurons()) < 4)
                   or (sum(v > 0 for v in edges)) < 4)
            else 0
        )
    )


def _evaluate(genome: Genome, fn: Callable[[dict, Simulation], Any],
              details: bool, monitor_path: Optional[Path] = None):

    monitor = (details and monitor_path is not None)
    controller = Brain(genome, robot, labels=monitor)
    data = controller_data(controller)

    if monitor:
        monitor_path.mkdir(exist_ok=True, parents=True)
        with open(monitor_path.joinpath("genome.dna"), "w") as f:
            json.dump(genome.to_json(), f)
        genome.to_dot(genome_data, monitor_path.joinpath("genome.png"), debug="keepdot", math="math.png")
        controller.ann.render3D(controller.labels).write_html(monitor_path.joinpath("ann.html"))

    for maze in mazes:
        controller.reset()
        if monitor:
            controller.monitor(monitor_path.joinpath(maze.to_string()))
        simulation = Simulation(maze, robot, save_trajectory=details)
        simulation.run(controller)
        if monitor:
            controller.stop_monitor()
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
    threads=os.cpu_count()-2,
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

genome_data = Genome.Data.create_for_eshn_cppn(
  dimension=3, seed=seed,
  with_input_bias=True, with_input_length=True,
  with_leo=True, with_output_bias=True,
  with_innovations=True, with_lineage=True
)

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


def trajectory(final: bool):
    _id = f"{evolver.generation:0{gen_digits}d}"
    merge_trajectories(
        list(_evaluate(evolver.champion.genome,
                       lambda _, s: MazeWidget.plot_trajectory(s, 500),
                       details=True,
                       monitor_path=path.joinpath(_id) if final else None)),
        trajectories_path.joinpath(f"{_id}.png"))


print(f"{optimal_fitness=}")

with evolver:
    start_time = time.perf_counter()

    trajectory(final=False)
    for g in range(gen):
        prev_best = evolver.best_fitness
        evolver.step()

        should_break = (evolver.best_fitness >= optimal_fitness)
        end = should_break or (g == gen-1)
        if (evolver.best_fitness > prev_best) or end:
            trajectory(final=end)

        if should_break:
            logging.warning("Optimal fitness reached")
            break

    print("Evolution finished in", humanize.precisedelta(timedelta(seconds=time.perf_counter() - start_time)))

evolver.generate_plots("png",
                       options=dict(optimal_fitness=optimal_fitness))
