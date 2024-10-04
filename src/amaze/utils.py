import logging
import math
from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter


def merge_trajectories(images: List[QImage], path: Path):
    n = len(images)
    if (n % 4) == 0:  # Assume we have all rotations (sequentially)
        images = [_merge(images[i:i+4]) for i in range(0, n, 4)]
        big_img = _merge(images, spacing=10)
    else:
        big_img = _merge(images)

    if not big_img.save(str(path)):
        logging.error(f"Could not save merged trajectories to {path}")

    # for i, img in enumerate(images):
    #     img.save(str(path.with_suffix(f".{i}.png")))


def _merge(images: List[QImage], spacing=0) -> QImage:
    nc = math.ceil(math.sqrt(len(images)))
    nr = math.ceil(len(images) / nc)

    ref = images[0]
    w, h = ref.width(), ref.height()

    big_img = QImage(nc * w + (nc - 1) * spacing,
                     nr * h + (nr - 1) * spacing,
                     QImage.Format_ARGB32)
    big_img.fill(Qt.transparent)
    painter = QPainter(big_img)

    for ix, img in enumerate(images):
        i, j = ix % nc, ix // nc
        x, y = i * (w + spacing), j * (h + spacing)
        painter.drawImage(x, y, img)

    painter.end()

    return big_img
