import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from abrain.core.ann import ANNMonitor
from amaze import Robot, OutputType, InputType, Maze
from amaze.simu import BaseController
from amaze.simu.pos import Vec
from amaze.simu.types import State

from abrain import Point3D, ANN3D
from abrain.core.genome import Genome


def create_genome_data(seed):
    return Genome.Data.create_for_eshn_cppn(
        dimension=3, seed=seed,
        with_input_bias=True, with_input_length=True,
        with_leo=True, with_output_bias=True,
        with_innovations=True, with_lineage=True
    )


def save(genome: Genome, robot: Robot.BuildData, path: Path | str):
    with open(path, "w") as f:
        j = genome.to_json()
        j["robot"] = robot.to_string()
        json.dump(j, f)


def load(path: Path):
    with open(path, "r") as gf:
        j = json.load(gf)
        if "robot" not in j:
            robot = Robot.BuildData.from_string("H7")
            logging.error("No robot specification in provided genome file. Is it an obsolete version?")
        else:
            robot = Robot.BuildData.from_string(j.pop("robot"))
        return Genome.from_json(j), robot


class Brain(BaseController):
    @staticmethod
    def generate_neurons(robot: Robot.BuildData, with_labels: bool):
        v = robot.vision
        inputs, outputs, labels = [], [], None
        if with_labels:
            labels = {}
        for j in range(v):
            for i in range(v):
                p = Point3D((2 * i / (v - 1) - 1),
                            -1,
                            1 - (2 * j / (v - 1)))
                inputs.append(p)
                if with_labels:
                    labels[p] = f"I[{i}, {j}]"

        outputs = [
            Point3D(+.5, 1, 0),
            Point3D(0, 1, +.5),
            Point3D(-.5, 1, 0),
            Point3D(0, 1, -.5),
        ]
        if with_labels:
            labels.update({p: d.name for p, d in zip(outputs, Maze.Direction)})

        return inputs, outputs, labels

    def __init__(self, genome: Genome, robot: Robot.BuildData,
                 labels: bool = False):
        super().__init__(robot)
        inputs, outputs, self.labels = self.generate_neurons(robot, labels)
        self.ann = ANN3D.build(inputs, outputs, genome)
        self._inputs, self._outputs = self.ann.buffers()

        self._monitor = None

    def reset(self):
        self.ann.reset()

    def monitor(self, folder: Path):
        if not folder.exists():
            folder.mkdir(parents=True)
        self._monitor = ANNMonitor(
            ann=self.ann, labels=self.labels,
            folder=folder,
            neurons_file="neurons.dat",
            dynamics_file="dynamics.dat",
            dt=1
        )

    def stop_monitor(self):
        self._monitor.close()

    @staticmethod
    def inputs_types() -> List[InputType]:
        return [InputType.CONTINUOUS]

    @staticmethod
    def outputs_types() -> List[OutputType]:
        return [OutputType.DISCRETE, OutputType.CONTINUOUS]

    def __call__(self, inputs: State) -> Vec:
        self._inputs[:] = inputs.flatten()
        # print([n.value for n in self.ann.output_neurons()])
        self.ann(self._inputs, self._outputs)
        # print(">", [n.value for n in self.ann.output_neurons()])
        if self._monitor is not None:
            self._monitor.step()
        return self.discrete_actions[np.argmax(self._outputs)]


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
