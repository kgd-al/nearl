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
# base = "M12_5x5"

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

df.drop("P_A", inplace=True)
# df.drop("MLP", inplace=True)
print("Refreshing MLP performance")


def gid(_genome): return _genome.stem


for genome in genomes:
    _gid = gid(genome)
    print("="*80)
    unseen_mazes = list(filter(lambda m: (_gid, maze_id(m)) not in df.index, mazes))

    output_base, rewards = process(genome, unseen_mazes, options)
    ddict = defaultdict(lambda: 0)
    for maze, reward in zip(unseen_mazes, rewards or []):
        ddict[maze_id(maze)] += (reward == 1)
    for key, value in ddict.items():
        df.loc[(_gid, key), "Reward"] = value

    merge_trajectories([
        QImage(str(output_base.joinpath(maze.to_string()).joinpath("trajectory.png")))
        for maze in mazes
    ], output_base.with_suffix(".trajectories.png"))

    print("-"*80)

df.sort_index(inplace=True)
df.to_csv(cache_file)

# Condensed form
margin = "%"
df = df.reset_index().pivot(index="Maze", columns="Genome")
sorter = np.vectorize(lambda _i: list(mazes_desc.keys()).index(_i))
df.sort_index(inplace=True, key=sorter)

df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([["Signs"], ["C", "T", "L"]]),
        data=[
            [int(stats[attr])
             for attr in ["clues", "traps", "lures"]
             if (stats := m.stats())]
            for _m in df.index
            if _m[0] == "M" and (m := Maze.from_string(_m, warnings=False))
        ],
        index=df.index
    ).join(df)

df["%Tot"] = 100 * (df == 4).sum(axis=1) / len(genomes)
df["Details"] = list(mazes_desc.values())
df.loc["Total (%)", [("Reward", gid(_g)) for _g in genomes]] = (
    100 * (df == 4).sum(axis=0)) / len(df)

print(df.to_string(float_format="%g", na_rep=""))
with open(folder.joinpath("summary.tex"), "wt") as f:
    f.writelines([
        fr"\def\mazes{{{', '.join(f'{k}/{v}' for k, v in mazes_desc.items())}}}", "\n",
        fr"\def\genomes{{{', '.join(g.stem for g in genomes)}}}", "\n",
        r"\def\circ#1{", "\n",
        r" \begin{tikzpicture}", "\n",
        r"  \pgfmathsetmacro{\a}{360*#1/4}", "\n",
        r"  \fill[red!10] (0, 0) circle (.4em);", "\n",
        r"  \fill[green] (.4em, 0) arc (0:\a:.4em) -- (0, 0) -- cycle;", "\n",
        r" \end{tikzpicture}", "\n",
        r"}", "\n"
    ])

    def formatter(x): return fr"\circ{{{int(x)}}}" if x <= 4 else x

    f.write(df.to_latex(escape=True, float_format="%g", na_rep="",
                        formatters={
                            ("Reward", gid(_g)): formatter
                            for _g in genomes
                        }))
