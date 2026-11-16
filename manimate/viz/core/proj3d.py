"""2.5D: 3-D pictures drawn as depth-sorted 2-D shapes, so no ThreeDScene is needed (README section 5.4).

World axes: x to the right, y into the screen, z up. ``depth`` grows with distance from the viewer.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from manim import ORIGIN, Create, Polygon, Text, UpdateFromAlphaFunc, VGroup, VMobject

from .palette import PALETTE, colormap

MAX_SURFACE_RES = 32  # performance budget, README section 14.2


class Projector3D:
    """Intuition: a camera that only turns. Orthographic, so parallel lines stay parallel."""

    def __init__(self, yaw: float = -40.0, pitch: float = 30.0, scale: float = 1.0, origin=ORIGIN):
        self.yaw, self.pitch, self.scale, self.origin = yaw, pitch, scale, np.array(origin, dtype=float)

    def rotated(self, yaw: float) -> "Projector3D":
        return Projector3D(yaw, self.pitch, self.scale, self.origin)

    def project(self, points) -> np.ndarray:
        """(N, 3) or (3,) world points -> same-shape array of ``(screen_x, screen_y, depth)``."""
        p = np.asarray(points, dtype=float)
        flat = p.reshape(-1, 3)
        cy, sy = np.cos(np.radians(self.yaw)), np.sin(np.radians(self.yaw))
        cp, sp = np.cos(np.radians(self.pitch)), np.sin(np.radians(self.pitch))
        x1 = flat[:, 0] * cy - flat[:, 1] * sy
        y1 = flat[:, 0] * sy + flat[:, 1] * cy
        depth = y1 * cp - flat[:, 2] * sp
        up = y1 * sp + flat[:, 2] * cp
        out = np.stack([x1, up, depth], axis=1) * self.scale
        out[:, 0] += self.origin[0]
        out[:, 1] += self.origin[1]
        return out.reshape(p.shape)

    def screen(self, points) -> np.ndarray:
        """Like ``project`` but returns Manim points ``(x, y, 0)``."""
        q = self.project(points).reshape(-1, 3).copy()
        q[:, 2] = 0.0
        return q


class Surface2p5D(VGroup):
    """Intuition: terrain seen from a helicopter: height is colour, and you can fly around it."""

    def __init__(self, Z_or_fn, u_range=(-2.0, 2.0), v_range=(-2.0, 2.0), *, resolution=(24, 24),
                 projector: Projector3D | None = None, cmap: str = "viridis", wire: bool = False):
        super().__init__()
        nu, nv = resolution
        if not (2 <= nu <= MAX_SURFACE_RES and 2 <= nv <= MAX_SURFACE_RES):
            raise ValueError(f"resolution must be between 2 and {MAX_SURFACE_RES} in each direction")
        self.projector = projector or Projector3D()
        self.us, self.vs = np.linspace(*u_range, nu), np.linspace(*v_range, nv)
        self.cmap, self.wire = cmap, wire
        self.Z = self._heights(Z_or_fn)
        self.zmin, self.zmax = float(self.Z.min()), float(self.Z.max())
        if self.zmax == self.zmin:
            self.zmax = self.zmin + 1.0
        self._rebuild()

    def _heights(self, Z_or_fn) -> np.ndarray:
        if callable(Z_or_fn):
            U, V = np.meshgrid(self.us, self.vs)
            Z = np.vectorize(Z_or_fn)(U, V).astype(float)
        else:
            Z = np.asarray(Z_or_fn, dtype=float)
        if Z.shape != (len(self.vs), len(self.us)):
            raise ValueError(f"Z must have shape {(len(self.vs), len(self.us))} (rows = v, columns = u)")
        return np.nan_to_num(Z)

    def _rebuild(self, Z: np.ndarray | None = None, projector: Projector3D | None = None) -> None:
        Z = self.Z if Z is None else Z
        proj = projector or self.projector
        quads = []
        for i in range(len(self.vs) - 1):
            for j in range(len(self.us) - 1):
                corners = np.array([[self.us[j], self.vs[i], Z[i, j]], [self.us[j + 1], self.vs[i], Z[i, j + 1]],
                                    [self.us[j + 1], self.vs[i + 1], Z[i + 1, j + 1]], [self.us[j], self.vs[i + 1], Z[i + 1, j]]])
                pr = proj.project(corners)
                quads.append((pr[:, 2].mean(), pr, Z[i:i + 2, j:j + 2].mean()))
        quads.sort(key=lambda q: -q[0])  # far first, so near quads paint over them
        self.submobjects = []
        for _, pr, z in quads:
            col = colormap(self.cmap, z, self.zmin, self.zmax)
            pts = [(p[0], p[1], 0.0) for p in pr]
            self.add(Polygon(*pts, stroke_width=1 if self.wire else 0, stroke_color=PALETTE.bg, fill_color=col, fill_opacity=0.95))

    def animate_rotate(self, yaw: float, run_time: float = 2.0):
        y0 = self.projector.yaw
        def step(m, a):
            m._rebuild(projector=m.projector.rotated(y0 + (yaw - y0) * a))
        anim = UpdateFromAlphaFunc(self, step, run_time=run_time)
        self.projector = self.projector.rotated(yaw)
        return anim

    def animate_morph_to(self, Z2, run_time: float = 2.0):
        Z0, Z1 = self.Z, self._heights(Z2)
        self.Z = Z1
        return UpdateFromAlphaFunc(self, lambda m, a: m._rebuild(Z=Z0 + (Z1 - Z0) * a), run_time=run_time)


class Path3D(VGroup):
    """Intuition: the track of a ball moving through space, drawn through the same camera as the surface."""

    def __init__(self, points, projector: Projector3D | None = None, *, color: str = PALETTE.yellow, width: float = 4.0):
        super().__init__()
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError("points must be an (N, 3) array with N >= 2")
        self.projector = projector or Projector3D()
        self.add(VMobject(color=color, stroke_width=width).set_points_as_corners(self.projector.screen(pts)))

    def animate_draw(self, run_time: float = 2.0):
        return Create(self[0], run_time=run_time)


class Axes2p5D(VGroup):
    """Intuition: three arrows meeting at a corner, so the viewer knows which way is up."""

    def __init__(self, projector: Projector3D | None = None, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0),
                 z_range=(0.0, 2.0), labels=("x", "y", "z")):
        super().__init__()
        self.projector = projector or Projector3D()
        for (lo, hi), axis, name in zip((x_range, y_range, z_range), np.eye(3), labels):
            a, b = self.projector.screen([axis * lo, axis * hi])
            self.add(VMobject(color=PALETTE.muted, stroke_width=2).set_points_as_corners([a, b]))
            t = Text(name, font_size=20, color=PALETTE.muted).move_to(b + 0.25 * (b - a) / max(np.linalg.norm(b - a), 1e-6))
            self.add(t)


class Sphere2p5D(VGroup):
    """Intuition: a glass globe. The far side of each circle is dimmed so you can tell front from back."""

    def __init__(self, radius: float = 1.0, projector: Projector3D | None = None, *, n_lat: int = 6, n_lon: int = 12,
                 color: str = PALETTE.blue):
        super().__init__()
        self.radius, self.projector = radius, projector or Projector3D()
        t = np.linspace(0, 2 * np.pi, 73)
        circles = []
        for k in range(1, n_lat):
            phi = -np.pi / 2 + np.pi * k / n_lat
            circles.append(np.stack([radius * np.cos(phi) * np.cos(t), radius * np.cos(phi) * np.sin(t), np.full_like(t, radius * np.sin(phi))], 1))
        for k in range(n_lon // 2):
            lam = np.pi * k / (n_lon // 2)
            circles.append(np.stack([radius * np.cos(t) * np.cos(lam), radius * np.cos(t) * np.sin(lam), radius * np.sin(t)], 1))
        for c in circles:
            pr = self.projector.project(c)
            for far, op in ((True, 0.2), (False, 0.8)):
                run = []
                for p in pr:
                    if (p[2] > 0) == far:
                        run.append((p[0], p[1], 0.0))
                    elif len(run) > 1:
                        self.add(VMobject(color=color, stroke_width=1.5, stroke_opacity=op).set_points_as_corners(run)); run = []
                    else:
                        run = []
                if len(run) > 1:
                    self.add(VMobject(color=color, stroke_width=1.5, stroke_opacity=op).set_points_as_corners(run))
