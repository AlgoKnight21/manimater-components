"""Shared building blocks for every viz domain (README section 5)."""
from .palette import PALETTE, Palette, colormap, contrast_ratio, lerp_color, phase_to_color  # noqa: F401
from . import numeric  # noqa: F401
from .labels import Annotation, MathLabel, tex_to_plain  # noqa: F401
from .focus import FocusRing, Spotlight, fade_in_whole  # noqa: F401
from .glow import Glow, GlowDot  # noqa: F401
from .flow import MAX_PARTICLES, FlowField, ParticleField, PulseTrain, pulse_along  # noqa: F401
from .grids import HeatmapGrid, MatrixGrid  # noqa: F401
from .live import AreaBand, CurveMorph, LiveCurve, LiveValue, ScrollingScope, Slider  # noqa: F401
from .diagram import Block, Wire, orthogonal_route  # noqa: F401
from .proj3d import Axes2p5D, Path3D, Projector3D, Sphere2p5D, Surface2p5D  # noqa: F401
from .plumbing import DEFAULT_DURATION, Beats, RecordingCtx, VizScene, dry_run  # noqa: F401
