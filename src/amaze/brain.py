from pathlib import Path
from typing import List, Optional

import numpy as np
from abrain.core.ann import ANNMonitor
from amaze import Robot, OutputType, InputType
from amaze.simu import BaseController
from amaze.simu.pos import Vec
from amaze.simu.types import State

from abrain import Point3D, ANN3D
from abrain.core.genome import Genome


class Brain(BaseController):
    @staticmethod
    def inputs(robot: Robot.BuildData):
        v = robot.vision
        return [
            Point3D((1 - 2 * i / (v - 1)),
                    -1,
                    (1 - 2 * j / (v - 1)))
            for j in range(v)
            for i in range(v)
        ]

    @staticmethod
    def outputs():
        return [
            Point3D(+.5, 1, 0),
            Point3D(0, 1, +.5),
            Point3D(-.5, 1, 0),
            Point3D(0, 1, -.5),
        ]

    def __init__(self, genome: Genome, robot: Robot.BuildData,
                 labels: bool = False):
        super().__init__(robot)
        self.ann = ANN3D.build(self.inputs(robot), self.outputs(), genome)
        self.labels
        self._inputs, self._outputs = self.ann.buffers()

        self._monitor = None

    def reset(self):
        self.ann.reset()

    def monitor(self, folder: Path):
        self._monitor = ANNMonitor(
            self.ann,
        )

    @staticmethod
    def inputs_types() -> List[InputType]:
        return [InputType.CONTINUOUS]

    @staticmethod
    def outputs_types() -> List[OutputType]:
        return [OutputType.DISCRETE, OutputType.CONTINUOUS]

    def __call__(self, inputs: State) -> Vec:
        self._inputs[:] = inputs.flatten()
        self.ann(self._inputs, self._outputs)
        return self.discrete_actions[np.argmax(self._outputs)]
