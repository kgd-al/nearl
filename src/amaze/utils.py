import math
from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter


def merge_plot(images: List[QImage], path: Path):
    nc = math.ceil(math.sqrt(len(images)))
    nr = math.ceil(len(images) / nc)

    ref = images[0]
    w, h = ref.width(), ref.height()

    big_img = QImage(nc * w, nr * h, ref.format())
    big_img.fill(Qt.yellow)
    painter = QPainter(big_img)

    print(f"{nc=} {nr=}")
    print([img.size() for img in images])
    print(f"{big_img.size()}")
    for ix, img in enumerate(images):
        x = w * (ix % nc)
        y = h * (ix // nc)
        print(ix, x, y)
        painter.drawImage(x, y, img)

    painter.end()
    big_img.save(str(path))
