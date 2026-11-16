#!/usr/bin/env bash
# Install the community Manim plugins that components/physics.py, ml.py, datastructures.py, chemistry.py,
# databases.py, circuits.py and extensions.py wrap.
# Run it after `pip install -r requirements.txt`. The Dockerfile and CI call it too.
#
# manim-physics and manim-ml are installed with --no-deps: their metadata pins an old manim (manim-physics needs
# <0.19), but both were checked to work on the manim this renderer uses. Their real dependencies are listed here.
set -euo pipefail
PIP="${PIP:-pip}"
$PIP install --no-cache-dir "pymunk>=6.6,<7" "shapely>=2,<3"
$PIP install --no-cache-dir --no-deps manim-physics==0.4.0 manim-ml==0.0.24
$PIP install --no-cache-dir manim-dsa==0.4.0 manim-chemistry==0.5.1 manim-eng==0.1.0
# manim-extensions (economics, mind maps, gears, queues...; used through components/_ext.py). Its own dependencies are light
# (manim, numpy, typing_extensions); segno and xmltodict are the small extras for QR codes and automata.
$PIP install --no-cache-dir manim-extensions==1.0.5 segno xmltodict
# manim-databases is early-alpha and not on PyPI, so it is installed from a pinned commit (MIT, depends only on manim).
$PIP install --no-cache-dir --no-deps "https://github.com/pg-pilot/manim-databases/archive/dee0ae1099444f4dabb115ea8041fa34f464effa.zip"
