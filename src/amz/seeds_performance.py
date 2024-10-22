#!/usr/bin/env python3

from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtGui import QImage

from amaze import Maze, StartLocation, MazeWidget, qt_application
from tabulate import tabulate

from utils import merge_trajectories

from rerun import process

base = "M8_6x6"

mazes_desc = {
    f"{base}_U": "Trivial",
    f"{base}_U_l0.25_L1": "Trivial (attractive lures)",
    f"{base}_U_l0.25_Lwarning-1": "Trivial (repulsive lures)",
    f"{base}_U_l0.25_Lwarning-.5": "Trivial (repulsive gray lures)",

    f"{base}_C1": "Simple (attractive)",
    f"{base}_t1_Twarning-1": "Inverted (attractive)",
    f"{base}_Cwarning-1": "Simple (repulsive)",
    f"{base}_t1_T1": "Inverted (repulsive)",

    f"{base}_C1_t0.5_T.5": "Trap (attractive)",
    f"{base}_Cwarning-1_t0.5_Twarning-.5": "Trap (repulsive)",
    f"{base}_C1_t0.5_Twarning-.5": "Trap (arr)",
    f"{base}_Cwarning-1_t0.5_T.5": "Trap (raa)",

    f"{base}_C1_l0.5_L.25_t0.5_T.5": "Complex (attractive)",
    f"{base}_Cwarning-1_l0.5_Lwarning-.25_t0.5_Twarning-.5": "Complex (repulsive)",
    f"{base}_C1_l0.5_Lwarning-.25_t0.5_Twarning-.5": "Complex (arr)",
    f"{base}_Cwarning-1_l0.5_L.25_t0.5_T.5": "Complex (raa)",
}
mazes = [_m
         for m in mazes_desc
         for _m in Maze.from_string(m, warnings=False).all_rotations()]


def maze_id(_m: Maze):
    return _m.build_data().where(start=StartLocation.SOUTH_WEST).to_string()


genomes = list(Path("saved/genome_seeds/").glob("*.dot"))

folder = Path("overleaf/seeds_performance/")
folder.mkdir(exist_ok=True, parents=True)

mazes_dir = folder.joinpath("mazes")
mazes_dir.mkdir(exist_ok=True, parents=True)
_ = qt_application()
for m in mazes_desc:
    if not (f := mazes_dir.joinpath(f"{m}.png")).exists():
        MazeWidget.static_render_to_file(Maze.from_string(m, warnings=False), f, size=512)

options = Namespace(
    folder=folder,
    monitor=False, trajectory=1, merge_trajectories=False,
    render=".pdf", render_3D=True, math=".math.pdf",
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

df.sort_index(inplace=True)
df.to_csv(cache_file)

# Condensed form
df = df.reset_index().pivot(index="Maze", columns="Genome")
df.sort_index(inplace=True, key=np.vectorize(lambda _i: list(mazes_desc.keys()).index(_i)))
print(df)
print()

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
df["Details"] = mazes_desc.values()

print(df.to_string(float_format=lambda x: f"{x:.1g}"))
with open(folder.joinpath("summary.tex"), "wt") as f:
    f.writelines([
        fr"\def\mazes{{{', '.join(f'{k}/{v}' for k, v in mazes_desc.items())}}}", "\n",
        fr"\def\genomes{{{', '.join(g.stem for g in genomes)}}}", "\n",
        r"\def\circ#1{\tikz\fill[#1] (0, 0) circle (.4em);}", "\n"
        r"\def\ok{\circ{green}}", "\n",
        r"\def\nok{\circ{red, opacity=.2}}", "\n"
    ])
    f.write(df.to_latex(float_format=lambda x: r"\ok" if x == 1 else r"\nok"))
