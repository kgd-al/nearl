#!/usr/bin/env python3

from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PyQt5.QtGui import QImage

from amaze import Maze, StartLocation
from tabulate import tabulate

from utils import merge_trajectories

from rerun import process

mazes = [
    "M3_5x5_U",
    "M3_5x5_U_l.5_L1",
    "M12_5x5_C1", "M12_5x5_t1_Twarning-1",
    "M12_5x5_Cwarning-1", "M12_5x5_t1_T1",
]
mazes = [_m
         for m in mazes
         for _m in Maze.from_string(m, warnings=False).all_rotations()]


def maze_id(_m: Maze):
    return _m.build_data().where(start=StartLocation.SOUTH_WEST).to_string()


genomes = Path("saved/genome_seeds/").glob("*.dot")

folder = Path("tmp/seeds_performance/")
folder.mkdir(exist_ok=True, parents=True)

options = Namespace(
    folder=folder,
    monitor=False, trajectory=1, merge_trajectories=False,
    render=".png", render_3D=True, math="png",
)

cache_file = folder.joinpath("cache.csv")
try:
    df = pd.read_csv(cache_file, index_col=[0, 1])
except:
    df = pd.DataFrame(columns=["Reward"],
                      index=pd.MultiIndex.from_product(
                          [[], []], names=["Genome", "Maze"]))
    print(df)

for genome in genomes:
    gid = genome.stem.split("_")[-1]
    print("="*80)
    unseen_mazes = list(filter(lambda m: (gid, maze_id(m)) not in df.index, mazes))

    output_base, rewards = process(genome, unseen_mazes, options)
    ddict = defaultdict(lambda: 0)
    for maze, reward in zip(unseen_mazes, rewards or []):
        ddict[maze_id(maze)] += reward
    for key, value in ddict.items():
        df.loc[(gid, key), "Reward"] = value / 4

    merge_trajectories([
        QImage(str(output_base.joinpath(maze.to_string()).joinpath("trajectory.png")))
        for maze in mazes
    ], output_base.with_suffix(".trajectories.png"))

    print("-"*80)

print(df)
df.sort_index(inplace=True)
df.to_csv(cache_file)

# Condensed form
df = df.reset_index().pivot(index="Maze", columns="Genome")
print(df)

df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([["Signs"], ["C", "T", "L"]]),
        data=[
            [stats[attr]
             for attr in ["clues", "traps", "lures"]
             if (stats := m.stats())]
            for _m in df.index
            if (m := Maze.from_string(_m, warnings=False))
        ],
        index=df.index
    ).join(df)


def fmt(x): return r"\ok" if x == 1 else r"\nok"


print(df.to_string(float_format=fmt))
print(df.to_latex(float_format=fmt))
