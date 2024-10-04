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
        self.ann(self._inputs, self._outputs)
        if self._monitor is not None:
            self._monitor.step()
        return self.discrete_actions[np.argmax(self._outputs)]
