#!/usr/bin/env python3

from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtGui import QImage
from abrain._cpp.phenotype import CPPN

from amaze import Maze, StartLocation, MazeWidget, qt_application, Sign
from amaze.misc.resources import np_images
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from tabulate import tabulate
from termcolor import colored

from utils import merge_trajectories

from rerun import process

if True:
    import matplotlib.pyplot as plt

    n = 51
    c = np.linspace(-1, 1, n)

    fns = CPPN.functions()
    bsgm = fns["bsgm"]
    step = fns["step"]
    sin = fns["sin"]
    abs = fns["abs"]

    outputs = [(.5, 0), (0, .5), (-.5, 0), (0, -.5)]

    plt.locator_params(nbins=5)

    def plot(_fn, title):

        data = [
            [[_fn(x, y, _x1, _y1) for x in c] for y in c]
            for _x1, _y1 in outputs
        ]

        vmin, vmax = np.quantile(data, [0, 1])
        vabs = max(vmin, -vmin, vmax, -vmax)
        vmin, vmax = -vabs, vabs

        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(2, 2, sharex="all", sharey="all")

        images = []
        for ax, data, (x1, y1) in zip(axes.flatten(), data, outputs):
            images.append(ax.imshow(data, aspect="equal", norm=norm,
                                    extent=(-1, 1, -1, 1), origin="lower",
                                    cmap="bwr"))
            ax.scatter([x1], [y1], c='k')

        fig.suptitle(title)
        fig.colorbar(images[0], ax=axes, orientation='vertical',
                     fraction=.1)

        for ax in axes[1, :]:
            ax.set_xlabel("X")
        for ax in axes[:, 0]:
            ax.set_ylabel("Y")

        return fig


    def fake_weight1(_x, _y, _x1, _y1):
        return bsgm(
                    -step(
                        step(
                            abs(_x+_x1)-1.4
                        )
                        +
                        step(
                            abs(_y+_y1)-1.4
                        )
                        - 0.5
                    )
                    + 0.1 * sin(
                        step(abs(_x) - .33)
                        + step(abs(_y) - .33)
                    )
        )

    def fake_weight2(_x, _y, _x1, _y1):
        return bsgm(
                    -step(
                        step(
                            abs(_x+_x1)-1.4
                        )
                        +
                        step(
                            abs(_y+_y1)-1.4
                        )
                        - 0.5
                    )
                    - 0.1 * sin(
                        step(abs(_x) - .33)
                        + step(abs(_y) - .33)
                    )
        )

    def fake_leo(_x, _y, _x1, _y1):
        return step(
            step(
                abs(.5*_x+_x1)
                - .7
            )
            +
            step(
                abs(.5*_y+_y1)
                - .7
            )
            - 0.5
        )

    with PdfPages("foo.pdf") as pdf:
        pdf.savefig(plot(fake_weight1, 'Weights (full)'))
        pdf.savefig(plot(fake_weight2, 'Weights (hollow)'))
        pdf.savefig(plot(fake_leo, 'LEO'))
        pdf.savefig(plot(lambda *args: fake_weight1(*args) * fake_leo(*args),
                         'Full'))
        pdf.savefig(plot(lambda *args: fake_weight2(*args) * fake_leo(*args),
                         'Hollow'))
    print("Generated plots")

    # exit(42)

# Seems to be working fine
# bsgm(-step(step(abs(x+x1)-1.4)+step(abs(y+y1)-1.4)-0.5)+0.1*kgdsin(step(abs(x)-.33)+step(abs(y)-.33)))
# bsgm(-step(step(abs(x+x1)-1.4)+step(abs(y+y1)-1.4)-0.5)-0.2*kgdsin(step(abs(x)-.33)+step(abs(y)-.33)))

# Connectivity
# step(step(abs(.5*x+x1)-.5)+step(abs(.5*y+y1)-.5)-0.5)

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
    f"{base}_U_l0.25_L1": "Trivial (attractive lures)",
    f"{base}_U_l0.25_Lwarning-1": "Trivial (repulsive lures)",
    f"{base}_U_l0.25_Lwarning-.5": "Trivial (repulsive gray lures)",
}
mazes_desc.update({
    f"{base}_C{c}-1_t.5_T{t}-1": l
    for c, t, l in [
        ("arrow", "warning", "Attractive full"),
        ("warning", "arrow", "Repulsive full"),
        ("rarrow", "alien", "Attractive hollow"),
        ("alien", "rarrow", "Repulsive hollow")
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

# df.drop("P_A", inplace=True)
# df.drop("MLP", inplace=True)
# print("Refreshing MLP performance")


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

    def formatter(x): return fr"\circ{{{int(x)}}}" if x <= 4 else x

    formatters = {("Signs", s): int for s in "CTL"}
    formatters.update({("Reward", gid(_g)): formatter for _g in genomes})
    f.write(df.to_latex(escape=True, float_format="%.1f", na_rep="",
                        formatters=formatters))
