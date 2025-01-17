#!/usr/bin/env python3
import json
from os.path import getmtime
from time import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtGui import QImage

from amaze import Maze, StartLocation, MazeWidget, qt_application
from rerun import process
from utils import merge_trajectories

# set pm3d; set hidden3d; set isosamples 50; set view equal xy; set xrange [-1:1]; set yrange [-1:1];
# gnuplot> set multiplot layout 2,4; do for [a in "-1 1"] { do for [an =0:270:90] { x1 = .5*cos(an); y1=.5*sin(an); set label 1 at x1, y1, 1.1 "" point pt 7 lc 'red' front; splot step(step(H10(x)-step(H11(y)-.45)-step(H10(-a*x)-.95)-.7)+step(H11(y)-step(H10(x)-.45)-step(H11(-a*y)-.95)-.7)-.5+step(step(H10(a*x)-.95)+step(H11(a*y)-.95))) title sprintf("a=%s, o=(%.g, %.g)", a, x1, y1); } }; unset multiplot;

if False:
    import cv2

    # Confirm that the signs have the expected characteristics
    errors = 0
    df = pd.DataFrame(columns=["AType", "Att", "FType", "Fll"])
    for s, a, f in [("arrow", "attractive", "full"),
                    ("warning", "repulsive", "full"),
                    ("rarrow", "attractive", "hollow"),
                    ("alien", "repulsive", "hollow")]:
        for r in list(range(5, 31, 2)) + [45, 125]:
            imgs = np_images([Sign(s, 1)], resolution=r)
            img = cv2.resize(imgs[0][0], dsize=(3, 3), interpolation=cv2.INTER_AREA)
            attractiveness = sum(img[:, 2] - img[:, 0])
            edge = -1 if a == "attractive" else 0
            fullness = img[1, edge] - .5*img[0, edge] - .5*img[2, edge]
            if (a == "attractive") != (attractiveness > 0):
                attractiveness = colored(attractiveness, "red")
                errors += 1
            if (f == "full") != (fullness > 0):
                fullness = colored(fullness, "red")
                errors += 1
            df.loc[f"{s}-{r}"] = [a, attractiveness, f, fullness]
            # print(s, r, attractiveness, fullness)
            # print(img)
            # print()
    print(tabulate(df, headers=df.columns))

    if errors > 0:
        print("Got some stuff wrong. Try again Bragg")
    exit(42)


base = "M8_6x6"
# base = "M12_5x5"

mazes_desc = {
    f"{base}_U": "Trivial",
}
mazes_desc.update({
    f"{base}_U_l0.25_Lstar-{v}": f"Trivial ({v} lures)"
    for v in [.25, .5, 1]
})
mazes_desc.update({
    f"{base}_U_l0.25_L{s}-1": f"Trivial ({s} lures)"
    for s in ["arrow", "warning", "rarrow", "alien"]
})
mazes_desc.update({
    f"{base}_C{c}-1_t.5_T{t}-1": l
    for c, t, l in [
        # ("arrow", "warning", "Attractive full"),
        # ("warning", "arrow", "Repulsive full"),
        # ("rarrow", "alien", "Attractive hollow"),
        # ("alien", "rarrow", "Repulsive hollow"),
        ("arrow", "rarrow", "Attractive full"),
        ("rarrow", "arrow", "Attractive hollow"),
        ("warning", "alien", "Repulsive full"),
        ("alien", "warning", "Repulsive hollow")
    ]
})
mazes_desc = {
    Maze.BuildData.from_string(k).to_string(): v
    for k, v in mazes_desc.items()
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
    retina=9, plot_cppn=True
)

cache_file = folder.joinpath("cache.csv")
try:
    df = pd.read_csv(cache_file, index_col=[0, 1])
    # raise ValueError()
except:
    df = pd.DataFrame(columns=["Reward"],
                      index=pd.MultiIndex.from_product(
                          [[], []], names=["Genome", "Maze"]))


def gid(_genome): return _genome.stem


with open(folder.joinpath("cache.json"), "a+") as cj:
    cj.seek(0)
    try:
        mtime_cache = json.load(cj)
    except Exception as e:
        mtime_cache = {}

    current_time = time()

    for genome in genomes:
        _gid = gid(genome)
        last_modif = getmtime(genome)
        last_eval = mtime_cache.get(_gid, 0)
        print(genome, last_modif, last_eval)

        if last_eval < last_modif and _gid in df.index:
            df.drop(_gid, inplace=True)
            print(f"Refreshing {_gid} performance")
            mtime_cache[_gid] = current_time

    cj.seek(0)
    cj.truncate()
    json.dump(mtime_cache, cj)

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

print(df.to_string(float_format=lambda x: f"{x:g}", na_rep=""))
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

    def formatter(x): return fr"\circ{{{int(x)}}}" if x <= 4 else f"{x:.3g}"

    formatters = {("Signs", s): int for s in "CTL"}
    formatters.update({("Reward", gid(_g)): formatter for _g in genomes})
    rows = df.to_latex(escape=True, float_format="%.1f", na_rep="",
                       formatters=formatters).split("\n")
    rows.insert(-8, r"\midrule")
    rows.insert(-4, r"\midrule")
    f.write("\n".join(rows) + "\n")
