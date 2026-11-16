# `manimate.viz`: the advanced visualization component library

> **Status: architecture proposal. First implementation: `viz.finance`** (8 of the section 10 scenes, self-contained: it does not yet use `viz.core`, which is still PR 1). The other domains are not started.
> Review it as a contract: every class, scene and utility named here is something a later PR must deliver, with tests.
> Section 15 lists the open questions that need a decision before code is written.

`manimate.viz` is a library of reusable Manim Mobjects, scene "beats" and animation utilities that explain advanced
technical topics in the style of 3Blue1Brown: geometric intuition first, smooth morphs, one idea per picture, and
motion that carries the meaning. It covers six domains:

| # | Domain | Package | One-line scope |
|---|--------|---------|----------------|
| 1 | Deep learning and generative AI | `viz.ml_advanced` | Attention, positional encoding, KV cache, diffusion, VAEs, loss landscapes, backprop |
| 2 | Microelectronics and computer architecture | `viz.microelectronics` | PN junction, MOSFET, CPU pipeline, caches, DMA, SPI/I2C/UART/USB/CAN |
| 3 | Control, signals, RF and DSP | `viz.signals` | Block diagrams, Bode/Nyquist, Fourier/Laplace, antennas, Smith chart, filters, FFT |
| 4 | Power systems, power electronics, machines | `viz.power` | Grid one-lines, buck/boost/H-bridge, 3-phase PWM, FOC, swing dynamics, faults |
| 5 | Quantitative finance | `viz.finance` | Greeks, GARCH, order book, yield curves, Brownian motion, binomial trees |
| 6 | Quantum computing and applied math | `viz.quantum_math` | Bloch sphere, circuits, phase portraits, eigen-decompositions |

The domain names in the original brief map to these packages as follows: "Advanced Deep Learning" is
`viz.ml_advanced` (the brief called it `manim_ml_advanced`); the other five are named after their field.

---

## Table of contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Fit with the existing repository](#2-fit-with-the-existing-repository)
3. [Architecture](#3-architecture)
4. [Component conventions](#4-component-conventions)
5. [Core library: `viz.core`](#5-core-library-vizcore)
6. [Domain 1: deep learning and generative AI](#6-domain-1-deep-learning-and-generative-ai-vizml_advanced)
7. [Domain 2: microelectronics and computer architecture](#7-domain-2-microelectronics-and-computer-architecture-vizmicroelectronics)
8. [Domain 3: control, signals, RF and DSP](#8-domain-3-control-signals-rf-and-dsp-vizsignals)
9. [Domain 4: power systems, power electronics, machines](#9-domain-4-power-systems-power-electronics-and-electrical-machines-vizpower)
10. [Domain 5: quantitative finance](#10-domain-5-quantitative-finance-vizfinance)
11. [Domain 6: quantum computing and applied math](#11-domain-6-quantum-computing-and-applied-math-vizquantum_math)
12. [Registry adapters and the catalog budget](#12-registry-adapters-and-the-catalog-budget)
13. [The 3Blue1Brown style guide, made testable](#13-the-3blue1brown-style-guide-made-testable)
14. [Testing, performance and delivery plan](#14-testing-performance-and-delivery-plan)
15. [Risks and open questions](#15-risks-and-open-questions)
16. [Appendix: traceability from the brief](#16-appendix-traceability-from-the-brief)

---

## 1. Goals and non-goals

**Goals**

- A modular set of primitives that can illustrate most advanced topics in the six domains, composed freely inside any
  standard Manim `Scene`.
- Every visual has a geometric or physical intuition behind it (matrix multiplication as space stretching,
  gradient descent as a ball rolling downhill, a depletion region as a charge-free moat).
- Every number on screen is **computed by a plain Python function** that can be unit-tested without Manim.
- Everything fits the product that already exists: components are registered in the same registry, appear in the same
  catalog, validate through the same storyboard path, and obey the same stage, theme and timing rules.

**Non-goals**

- No rewrite of the 44 components that already ship. Where an existing component overlaps (section 2.3), it stays;
  the new Mobjects may be extracted from it in a separate refactor PR.
- No LLM-written code at render time. Free-form input reaches the renderer only as JSON validated against the catalog,
  and expressions only through `safe_expr`.
- No new heavy dependencies. The library uses Manim, NumPy and SciPy (both already required by Manim), and the Python
  standard library. No PyTorch, no JAX, no plotting backends.
- Not a simulator. Models are chosen to be *right enough to teach* (a square-law MOSFET, a CRR binomial tree, a
  classical swing equation), and each component states its modelling assumption on screen or in its docstring.

---

## 2. Fit with the existing repository

This proposal was written against the repository as it is today. These facts constrain the design.

### 2.1 Constraints that shape the design

| Fact in the repo | Consequence for `viz` |
|------------------|-----------------------|
| A component is `fn(ctx: Ctx, **params)` registered with `@component` in `renderer/manimate/registry.py`; one declaration drives validation, the LLM catalog and docs. | Each scene is written once as a `play_*` function that takes a `Ctx` and is both a registry component and runnable standalone. |
| `components/README.md` design rules: one idea per component, **computed not drawn**, fade the finished frame in as a whole, fill narration time with `ctx.remaining`, stay on the 12.4 x 5.8 stage, theme colours only, at most 5 params, example = test. | These rules are adopted unchanged (section 13 turns them into automated checks). |
| The renderer's scene is a plain 2D `StoryboardScene(Scene)`; there is no `ThreeDScene` path. | 3D visuals use a **2.5D projection** (section 5, `Projector3D`) that works inside the existing scene. A native 3D path is an open question (section 15). |
| Registry param types are `string`, `number`, `integer`, `boolean`, `string[]`, `number[]`, `object[]`; **lists are capped at 8 items**, strings at 300 characters. | Registry adapters take small semantic parameters (presets, a few numbers). Large arrays (weight matrices, price histories) are generated from a `seed` and a preset. The Mobject layer itself accepts any NumPy array. |
| `safe_expr` evaluates `f(x)` only, with a fixed function whitelist. | Multi-variable expressions (`f(x, y)`, `dx/dt`, `dy/dt`) need a small additive extension to `safe_expr` (delivered in the core PR). Mobjects accept Python callables; adapters accept only presets or `safe_expr` strings. |
| The worker renders the **entire catalog as text into the planner prompt** (`catalogText` in `worker/src/prompt.ts`). The catalog is currently 44 components and about 36 KB (about 830 bytes per entry). `usableFor` only removes names a template lists in `avoid`. | Adding the adapters in section 12 would more than double the prompt. A domain-gated catalog is proposed in section 12.3. |
| `ctx.HAS_LATEX` exists because LaTeX may be absent. | All maths labels go through `MathLabel`, which falls back to plain text. |
| Scenes are kept under about 20 seconds; the planner writes 5 to 7 scenes per video. | A rich topic is several short scenes, selected by a `view` parameter on one adapter, not one long scene. |
| `theme.py` holds the palette (`BG` black, `ACCENT` blue, `YELLOW`, `GOOD`, `WARN`, `DANGER`, `PURPLE`). | `viz.core.palette` extends it (adds `TEAL`, colormaps); it never redefines existing names. |
| CI runs `pytest` (fast), `pytest -m slow` (real renders) and diffs `python -m manimate.catalog` against `worker/src/catalog.json`. | Every PR in this series regenerates the catalog and ships fast tests plus at least one `slow` render. |

### 2.2 Where things live

```
renderer/manimate/
├── viz/                      # NEW: this library (this README lives here)
│   ├── README.md
│   ├── core/                 # shared Mobjects, updaters, colormaps, 2.5D projection, numeric kernels
│   ├── ml_advanced/          # one subpackage per domain
│   ├── microelectronics/
│   ├── signals/
│   ├── power/
│   ├── finance/
│   └── quantum_math/
├── components/               # existing packs, plus NEW thin adapters: adv_ml.py, adv_micro.py, ...
└── ...
renderer/examples/viz/        # NEW: standalone example scenes and JSON storyboards
renderer/tests/viz/           # NEW: tests mirroring the package layout
```

`setuptools` already includes `manimate*`, so the new subpackages ship in the renderer image with no packaging change.

### 2.3 Overlap with what already exists

Names below are already taken (registry) or reserved (roadmap in `components/README.md`).

| Existing / reserved | Status | What this series does |
|--------------------|--------|-----------------------|
| `neural_network` (wraps ManimML) | shipped | Stays. `BackpropNetwork` is a separate, dependency-free Mobject because backprop needs per-edge control the plugin does not expose. |
| `attention_heatmap` (roadmap #50) | reserved | **Adopted** by this series as an adapter over `AttentionHeatmap` and `AttentionArcs`. |
| `gradient_descent` (roadmap #46) | reserved | **Adopted** as an adapter over `LossLandscape` and `DescentPath`. |
| `bode_plot` (roadmap #32) | reserved | **Adopted** as an adapter over `BodePlot`. |
| `linear_transformation` (roadmap #17) | reserved | **Adopted** as an adapter over `TransformableGrid`. |
| `ac_waveform`, `phasor_diagram`, `rotating_field`, `synchronous_motor`, `transformer`, `power_triangle`, `torque_speed_curve`, `circuit`, `circuit_diagram` | shipped | Stay unchanged. New power Mobjects reuse their visual language; where a drawing helper is worth sharing it is extracted in its own refactor PR, never inside a feature PR. |
| `logic_gates` (#30), `normal_distribution` (#19), `rc_charging_curve` (#31), `unit_circle` (#14) | reserved | Out of scope for this series; untouched. |

The follow-up to this README (not part of it) marks the four adopted roadmap rows as "owned by the viz series".

---

## 3. Architecture

Three layers, each depending only on the one beneath it.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Layer 3  Registry adapters   components/adv_*.py                            │
│          @component(...) functions: tiny params, presets, plain-English     │
│          validation. They build nothing themselves; they call Layer 2.      │
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 2  Domain library      viz/<domain>/                                  │
│          M  Mobjects (VGroup subclasses): build the static frame            │
│          S  Scene beats  play_*(ctx, ...): the animation script             │
│          P  Pure math (no Manim import): every number the picture shows     │
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 1  Core                viz/core/                                      │
│          palette, MathLabel, glow, flow pulses, heatmap, matrix grid,       │
│          particles, wires, 2.5D projection, live curves, readouts, timing   │
└────────────────────────────────────────────────────────────────────────────┘
```

**Why three layers.** The product needs small, forgiving, LLM-friendly parameters (Layer 3). Authors composing their own
scenes need rich, array-accepting Mobjects (Layer 2). Both need the same visual vocabulary so that every video looks like
one series (Layer 1). Keeping the layers separate means the catalog can stay small while the Mobject library grows.

**One script, two ways to run it.** A scene is a function `play_x(ctx, **params)`. The registry adapter registers it for
storyboards. `VizScene` (core) wraps the same function in a standalone Manim `Scene` for the example scripts:

```python
# renderer/examples/viz/attention_head.py   ->   manim -ql attention_head.py AttentionHeadExample
from manimate.viz.core import VizScene
from manimate.viz.ml_advanced import play_attention_head

class AttentionHeadExample(VizScene):
    def play(self, ctx):
        play_attention_head(ctx, tokens=["The", "cat", "sat", "down"], query_index=3, causal=True)
```

So there is a single source of truth for each animation, and the example script cannot drift from the product.

**Kinds used in the tables below**

| Tag | Meaning |
|-----|---------|
| **M** | Mobject, a `VGroup` subclass. Builds its full static picture in `__init__`. Never touches `Scene` or `Ctx`. |
| **S** | Scene beats function `play_*(ctx, ...)`, registered as a component and runnable via `VizScene`. |
| **U** | Animation or updater utility: a function that returns an `Animation` or attaches an updater. |
| **P** | Pure math function or dataclass. Does not import Manim. Unit-tested directly. |

---

## 4. Component conventions

These apply to every Mobject, scene and utility in the library.

### 4.1 Mobject contract

```python
class AttentionHeatmap(VGroup):
    def __init__(self, Q, K, *, tokens=None, causal=False, scale=True, cmap="viridis", palette=PALETTE): ...

    # 1. The static frame is complete after __init__ (it can fade in as a whole).
    # 2. animate_* methods RETURN an Animation; they never call scene.play().
    def animate_dot_product(self, i: int, j: int, *, run_time: float = 1.2) -> Animation: ...
    def animate_softmax(self, *, run_time: float = 1.0) -> Animation: ...

    # 3. set_* mutate immediately (used before the first frame or inside updaters).
    def set_values(self, scores) -> Self: ...

    # 4. bind() wires a ValueTracker to a continuous parameter via an updater (the "slider" behaviour).
    def bind(self, tracker: ValueTracker, param: str) -> Self: ...

    # 5. .data exposes the pure-math result (a frozen dataclass) so tests and narration can read exact numbers.
    @property
    def data(self) -> AttentionData: ...
```

### 4.2 Rules

1. **Computed, not drawn.** Each Mobject takes its numbers from a function in the domain's `_math.py` (kind **P**).
   Tests assert on those numbers first and on geometry second.
2. **Deterministic.** Anything random takes `seed: int = 7` and uses `numpy.random.default_rng(seed)`. Two renders
   of the same storyboard are identical.
3. **Fade in whole.** The static frame appears with one `FadeIn` (`fade_in_whole`, about 1.2 to 1.5 s); only then do the parts
   that carry the meaning animate. Part-by-part construction of a static diagram is not allowed.
4. **Fill the narration time.** Beats finish with `ctx.finish()` or a final animation sized from `ctx.remaining`;
   reveals share time through `Ctx.beat`.
5. **Stay on the stage.** About 12.4 x 5.8 units; the caption owns the bottom. Mobjects are placed with `ctx.fit` (or `fit` from
   `components/_plugin.py` when scaling *up* is needed). Nothing is hard-coded to the 16:9 frame edge.
6. **Theme colours only.** Colours come from `viz.core.palette`. One accent per picture marks "look here", and colour is
   never the only carrier of meaning: pair it with a label, a position or a shape.
7. **Labels are short plain words.** Under 28 characters where possible, wrapped with `label()`. Maths goes through `MathLabel`.
8. **Graceful limits.** Out-of-range input is clamped with a plain-English note (adapters) or raises `ValueError` with a
   specific message (Mobjects). Mobject counts are capped per component (section 14.2); above the cap the component
   switches to an image-based or aggregated rendering instead of slowing down.
9. **No mutation of Manim globals.** No monkey-patching `config`, `AnimationGroup` or `Text` except through the existing
   helpers in `components/_plugin.py`.
10. **Independent originality.** Implemented from scratch for this repository. No code is copied from other video
    projects or tutorials; any snippet adopted from elsewhere must have its licence checked first.

### 4.3 Registry adapter contract

- At most 5 params, each with a default or a plain-English range error; lists of at most 8 items.
- A `view` or `variant` parameter with `choices` selects between related pictures. This is what keeps the catalog small.
- `example` must render; each adapter has a fast dry-run test and one `slow` render.
- Domain tag is one of `machine-learning`, `microelectronics`, `signals`, `electrical`, `finance`, `quantum`, `math`
  (existing tags are reused where a domain already exists).

---

## 5. Core library: `viz.core`

Everything here is shared by two or more domains. It ships first (PR 1) because the domains depend on it.

### 5.1 Palette, labels, focus

| Name | Kind | Signature (key params) | Purpose |
|------|------|------------------------|---------|
| `PALETTE` | P | `PALETTE.blue / yellow / teal / red / green / purple / muted / fg` | Extends `theme`. Blue = primary signal, yellow = derived quantity or "look here", teal = structure and secondary signal, red = error, gradient and fault, green = success and hit. Adds `TEAL`. |
| `colormap` | P | `colormap(name, v, vmin, vmax) -> hex`; names: `viridis`, `magma`, `coolwarm`, `blue_yellow` | Sequential and diverging maps with a black-friendly low end. Also `lerp_color`, `phase_to_color(theta)` (hue = phase, used for complex amplitudes). |
| `MathLabel` | M | `MathLabel(tex, fallback=None, size=28, color=FG)` | `MathTex` when `ctx.HAS_LATEX`, otherwise `Text(fallback or tex_to_plain(tex))`. Mandatory for every formula. |
| `Annotation` | M | `Annotation(text, target, direction=UP, arrow=True)` | Short label with an arrow to a target. Keeps text off the data. |
| `FocusRing` | U | `FocusRing(target, color=YELLOW, pulse=True)` | The one "look here" device. |
| `Spotlight` | U | `Spotlight(target, dim=0.25)` | Dims everything except the target (`animate_on` / `animate_off`). |
| `fade_in_whole` | U | `fade_in_whole(ctx, mob, run_time=1.3)` | Implements rule 4.2.3. |

### 5.2 Light, flow and particles

| Name | Kind | Signature | Purpose |
|------|------|-----------|---------|
| `Glow` | M | `Glow(mobject, color, layers=4, spread=0.12)` | Cheap glow from stacked low-opacity strokes. No shaders, so it works in the Cairo renderer. |
| `GlowDot` | M | `GlowDot(point, radius, color)` | A glowing point: electrons, pulse heads, state-vector tips. |
| `pulse_along` | U | `pulse_along(path, color, run_time, width=0.12, trail=0.25) -> Animation` | A glowing pulse travelling a path. The primitive behind signals, current and gradients. |
| `PulseTrain` | U | `PulseTrain(paths, delays, color, reverse=False)` | Several pulses with staggered start, optionally reversed (backprop). |
| `FlowField` | M | `FlowField(path, n=24, speed=1.0, color)` | Continuous particles drifting along a path via an updater (current, flow of charge, data on a bus). |
| `ParticleField` | M | `ParticleField(n, bounds, kind="dot", drift=None, seed=7)` | Seeded particle cloud with per-frame drift and a `.remove(ids)` / `.recombine(pairs)` animation. Used for carriers, noise, samples. |

### 5.3 Data pictures shared across domains

| Name | Kind | Signature | Purpose |
|------|------|-----------|---------|
| `HeatmapGrid` | M | `HeatmapGrid(values, cmap="viridis", vmin=None, vmax=None, cell=0.4, row_labels=None, col_labels=None, show_values=False)` | Coloured grid. `.animate_values(new)`, `.highlight_cell(i, j)`, `.highlight_row(i)`. Above 16 x 16 cells it switches to one `ImageMobject`. |
| `MatrixGrid` | M | `MatrixGrid(values, name=None, row_labels=None, col_labels=None, decimals=2)` | Labelled numeric matrix with a shape tag. `.row(i)`, `.col(j)`, `.animate_multiply(other)` (row-by-column sweep). |
| `LiveCurve` | M | `LiveCurve(axes, fn, tracker=None, x_range=None, color)` | A curve re-evaluated every frame from `fn(x, t)`, so a slider morphs it continuously. |
| `AreaBand` | M | `AreaBand(axes, lower_fn, upper_fn, color, opacity=0.2)` | Shaded band between two functions: variance bands, SWR envelopes, confidence fans. |
| `CurveMorph` | U | `CurveMorph(curve_a, curve_b, run_time)` | Morph between two sampled curves with matched point counts (stable under `Transform`). |
| `ScrollingScope` | M | `ScrollingScope(signal_fn, window=4.0, tracker, color)` | A strip chart that scrolls as time advances. |
| `Slider` | M | `Slider(tracker, value_range, label, unit="", decimals=2)` | A read-only slider whose knob follows a `ValueTracker`. The video analogue of an interactive control: the picture is parameter-driven and the narrator "drags" it by animating the tracker. |
| `LiveValue` | M | `LiveValue(label, fn_or_tracker, unit="", decimals=2)` | Numeric readout that updates every frame. |
| `Wire` | M | `Wire(points, arrow=False, color, width=3)` + `orthogonal_route(a, b, via=None)` | Orthogonal connectors for block diagrams, schematics and buses. |
| `Block` | M | `Block(label, w=1.6, h=0.9, color)` | Labelled rectangle with named ports (`left`, `right`, `top`, `bottom`). |

### 5.4 2.5D projection (the 3D strategy)

The renderer uses a plain `Scene`, so "3D" means a lightweight projection that draws filled quads and lines in 2D, depth-sorted.

| Name | Kind | Signature | Purpose |
|------|------|-----------|---------|
| `Projector3D` | P | `Projector3D(yaw=-40, pitch=30, scale=1.0, origin=ORIGIN)` | Orthographic projection `(x, y, z) -> (screen_x, screen_y, depth)`. `.project(points)` is vectorised. |
| `Surface2p5D` | M | `Surface2p5D(Z_or_fn, u_range, v_range, resolution=(24, 24), projector, cmap, wire=False)` | Depth-sorted quad mesh coloured by height. `.animate_rotate(yaw)`, `.animate_morph_to(Z2)`. Resolution is capped at 32 x 32. |
| `Path3D` | M | `Path3D(points, projector, color, trail=True)` | A 3D polyline or moving point drawn through the projector; always depth-tested against a `Surface2p5D`. |
| `Axes2p5D` | M | `Axes2p5D(projector, x_range, y_range, z_range, labels)` | Three axes with ticks and labels in the same projection. |
| `Sphere2p5D` | M | `Sphere2p5D(radius, projector, n_lat=6, n_lon=12)` | Wireframe sphere with depth-shaded great circles (base of the Bloch sphere). |

### 5.5 Timing and scene plumbing

| Name | Kind | Signature | Purpose |
|------|------|-----------|---------|
| `Beats` | U | `Beats(ctx, n, cap=1.0, floor=0.3)` | Hands out per-reveal durations from `ctx.remaining` (wraps `Ctx.beat`). |
| `VizScene` | M | `class VizScene(Scene)` with `play(self, ctx)` | Standalone runner: black background, a `Ctx` with a generous duration, so example scripts and tests run any `play_*` function unchanged. |
| `numeric` | P | `softmax, layernorm, rk4, euler, gaussian, rng(seed), moving_average, normalize, safe_div` | Small NumPy kernels shared by the domains. |
| `safe_expr` extension | P | `compile_expr(text, variables=("x",)) -> Callable` | Additive change to `manimate/safe_expr.py`: allows named variables (`x`, `y`, `t`, `s`) for 2D landscapes and vector fields. Same whitelist, same length cap, same no-`eval` rule. |

### 5.6 Core usage

```python
from manim import DOWN, FadeIn, Scene, ValueTracker
from manimate.viz.core import HeatmapGrid, Slider

class Demo(Scene):
    def construct(self):
        t = ValueTracker(0)
        grid = HeatmapGrid([[0.1, 0.9], [0.5, 0.3]], cmap="viridis", row_labels=["a", "b"], col_labels=["x", "y"])
        slider = Slider(t, (0, 1), "mix").next_to(grid, DOWN)
        self.play(FadeIn(grid), FadeIn(slider))
        self.play(t.animate.set_value(1), grid.animate_values([[0.9, 0.1], [0.3, 0.5]]), run_time=2)
```

---

## 6. Domain 1: deep learning and generative AI (`viz.ml_advanced`)

**Intuition map.** A projection matrix *stretches and rotates* a token's space. A dot product is *how much two arrows
agree*. Softmax is *a budget of attention that must sum to 1*. Diffusion is *a cloud of data blurring into a Gaussian and a
model learning to run the film backward*. A loss landscape is *terrain*; an optimizer is *a ball with (or without) momentum*.
Backprop is *blame travelling backward along the same wires the prediction travelled forward*.

### 6.1 Pure math (`ml_advanced/_math.py`)

| Function | Returns |
|----------|---------|
| `qkv_project(X, Wq, Wk, Wv)` | `Q, K, V` |
| `attention_scores(Q, K, causal, scale)` / `attention_weights(...)` / `attention_output(...)` | raw scores, softmax weights, weighted sum of `V` |
| `sinusoidal_pe(n_pos, d_model, base=10000)` | positional encoding matrix; `pe_components(pos, d_model)` returns the individual sin/cos terms for the wave-summation view |
| `kv_cache_steps(n_tokens, window=None)` | per-step list of cached slots, evicted slots, and cells recomputed with and without a cache |
| `beta_schedule(kind, T)` / `alpha_bar(betas)` | linear or cosine noise schedule and cumulative signal level |
| `forward_noise(x0, t, schedule, seed)` | `x_t` and the noise `eps` used |
| `reverse_step(xt, eps_hat, t, schedule)` | one denoising step (DDPM-style mean) |
| `vae_reparam(mu, logvar, seed)` | sampled `z`; `kl_divergence(mu, logvar)` |
| `latent_interpolate(za, zb, n, mode="linear"\|"slerp")` | interpolation path |
| `optimizer_path(loss, grad, start, method, lr, steps, momentum, beta1, beta2)` | positions, gradient vectors and **velocity (momentum) vectors** per step for `sgd`, `momentum`, `rmsprop`, `adam` |
| `backprop_trace(layer_sizes, seed, activation)` | activations, pre-activations, and `dL/dw` per edge from a real forward and backward pass (tiny network, MSE loss) |

### 6.2 Transformer and LLM visuals

| Name | Kind | Signature (key params) | Shows / key methods |
|------|------|------------------------|---------------------|
| `TokenStrip` | M | `TokenStrip(tokens, embeddings=None, highlight=None)` | A row of token boxes, optionally with a tiny embedding bar under each. `.animate_highlight(i)`. |
| `QKVProjection` | M | `QKVProjection(X, d_head, seed=7, weights=None)` | Grids for `X`, `W_Q`, `W_K`, `W_V` and the resulting `Q`, `K`, `V`. `.animate_project()` sweeps each token row through the three matrices and drops the result into place (matrix multiplication as a change of space). |
| `AttentionHeatmap` | M | `AttentionHeatmap(Q, K, tokens=None, causal=False, scale=True, cmap="viridis")` | The score grid. `.animate_dot_product(i, j)` lights row `q_i` and column `k_j`, shows multiply-and-sum, and fills cell `(i, j)` live. `.animate_scale()`, `.animate_mask()` (grey-out of the future), `.animate_softmax()` (each row normalises to 1, bars equalise visibly). |
| `AttentionArcs` | M | `AttentionArcs(tokens, weights, query_index, top_k=None)` | Arcs from one token to the others, thickness and brightness proportional to weight. Pairs with the heatmap: same numbers, second view. |
| `AttentionHead` | M | `AttentionHead(tokens, d_head=4, causal=False, seed=7)` | The whole path: `X -> Q,K,V -> scores -> softmax -> weights x V -> output`. `.animate_forward(query_index)` walks one token through it. |
| `MultiHeadAttention` | M | `MultiHeadAttention(tokens, n_heads=3, d_head=4, seed=7)` | Small multiples of heads (each with its own pattern), then concatenation and output projection. |
| `PositionalEncodingWaves` | M | `PositionalEncodingWaves(d_model=8, n_positions=12, n_freqs=4)` | Stacked sine/cosine curves at geometrically spaced frequencies. `.animate_scan(position)` slides a vertical cursor and reads the values off into a vector. |
| `PositionalEncodingSum` | M | `PositionalEncodingSum(tokens, d_model=8)` | Embedding bars for each token **plus** the wave values overlaid, then summed into one bar set (wave summation onto the embedding). `.animate_add()`. |
| `PEHeatmap` | M | `PEHeatmap(n_positions, d_model)` | The classic positions x dimensions image, linked to the waves by a column highlight. |
| `KVCacheBuffer` | M | `KVCacheBuffer(capacity=8, window=None, show_values=False)` | Two rows of memory slots (K and V). `.animate_append(token)`; with `window` set, `.animate_slide()` evicts the oldest slot and shifts the window (sliding-window attention). `.highlight_attended(query_idx)`. |
| `DecodeTimeline` | M | `DecodeTimeline(tokens, use_cache=True, window=None)` | Autoregressive generation, one new token per step. Cells that must be recomputed are red; cells read from cache are teal. A running counter compares work with and without the cache. |

**Scenes (kind S):** `play_attention_head`, `play_attention_heatmap`, `play_positional_encoding`, `play_kv_cache`.

### 6.3 Latent space and diffusion

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `LatentGrid` | M | `LatentGrid(values_or_fn, shape=(12, 12), cmap="viridis")` | A 2D feature map as a coloured grid, small enough to morph cell by cell. Procedural images (shapes, gradients, glyphs) so no external files are needed. |
| `SchedulePlot` | M | `SchedulePlot(kind="cosine", T=100)` | `beta_t` and `alpha_bar_t` curves with a time cursor; shows how fast the signal is destroyed. |
| `ForwardDiffusion` | M | `ForwardDiffusion(x0, T=100, frames=(0, 10, 30, 60, 100), schedule="cosine", seed=7)` | A filmstrip `x_0 -> x_T` of `LatentGrid`s with Gaussian noise injected; `.animate_noise(t)` morphs the grid continuously. |
| `ReverseDiffusion` | M | `ReverseDiffusion(xT, eps_model=None, T=100, frames=..., seed=7)` | Noise morphing back into a clear feature map. Shows the predicted noise `eps_hat` as a second grid that is *subtracted* each step. `.animate_denoise(t_from, t_to)`. |
| `Distribution2DDiffusion` | M | `Distribution2DDiffusion(points, T=100, seed=7)` | A 2D data cloud (two moons, spiral, rings) dissolving into an isotropic Gaussian; with `ScoreField` it runs in reverse. The geometric picture of diffusion. |
| `ScoreField` | M | `ScoreField(density_fn_or_points, bounds, density=14)` | Arrows pointing toward higher data density (the score). |
| `Autoencoder` | M | `Autoencoder(layer_sizes=(8, 5, 2, 5, 8), labels=("encoder", "bottleneck", "decoder"))` | Hourglass network with a highlighted bottleneck. `.animate_encode()`, `.animate_decode()`. |
| `LatentSpace2D` | M | `LatentSpace2D(points, classes=None, bounds=(-3, 3), decoder=None)` | Latent scatter with class colours and decoded thumbnails; `.animate_probe(z)` decodes any point. |
| `LatentInterpolation` | M | `LatentInterpolation(space, za, zb, n=6, mode="linear")` | A curve through latent space, with the decoded output morphing along it (the "interactive interpolation curve"). `.animate_walk()`. |
| `VAESampling` | M | `VAESampling(mu, sigma)` | The reparameterisation trick: `z = mu + sigma * eps` as a shifted, scaled Gaussian cloud, with the KL term as the gap to the prior. |

**Scenes:** `play_diffusion_process` (`view`: forward, reverse, both; `variant`: image, points), `play_autoencoder_latent`.

### 6.4 Optimization and training

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `LossLandscape` | M | `LossLandscape(fn_or_preset, x_range, y_range, projector=None, resolution=(28, 28), show_contours=True)` | A 3D surface (via `Surface2p5D`) with an optional contour map inset. Presets: `bowl`, `ravine`, `saddle`, `two_minima`, `beale`. |
| `DescentPath` | M | `DescentPath(landscape, method="sgd", start=(-2, 1.5), lr=0.1, steps=40, momentum=0.9, betas=(0.9, 0.999))` | A ball rolling on the surface leaving a trail. Draws the **gradient arrow** (red) and the **momentum/velocity arrow** (yellow) at each step. `.animate_descent()`. |
| `OptimizerRace` | M | `OptimizerRace(landscape, methods=("sgd", "momentum", "adam"), start, lr)` | The same start for several optimizers side by side, with a loss-vs-step inset. |
| `LearningRateStrip` | M | `LearningRateStrip(landscape, lrs=(0.01, 0.1, 0.9))` | Too small, about right, too large (overshoot and divergence) on one landscape. |
| `LossCurve` | M | `LossCurve(losses, smoothing=0.0, tracker=None)` | Training curve that draws itself as the optimizer steps. |

**Scene:** `play_gradient_descent`.

### 6.5 Backpropagation

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `BackpropNetwork` | M | `BackpropNetwork(layer_sizes, activation="relu", seed=7)` | A fully drawn network (own drawing, no plugin). `.animate_forward()` sends blue pulses left to right; `.animate_backward()` sends **reversed glowing pulses** (yellow to red) carrying `dL/dw` labels; edge thickness and colour follow gradient magnitude. |
| `ChainRuleStrip` | M | `ChainRuleStrip(factors, target="dL/dw")` | The product of local derivatives; `.animate_chain()` lights each factor in the order the backward pulse reaches it. |
| `ComputationGraph` | M | `ComputationGraph(nodes, edges, values)` | A tiny scalar graph (for example `L = (wx - y)^2`) with forward values above nodes and backward gradients below. |

**Scene:** `play_backprop_flow`.

### 6.6 Usage

```python
from manimate.viz.ml_advanced import LossLandscape, DescentPath, OptimizerRace
from manimate.viz.core import Projector3D, fade_in_whole

land = LossLandscape("ravine", x_range=(-2, 2), y_range=(-1, 3), projector=Projector3D(yaw=-50, pitch=35))
race = OptimizerRace(land, methods=("sgd", "momentum", "adam"), start=(-1.6, 2.4), lr=0.05)

fade_in_whole(ctx, land)                       # the static terrain appears together
ctx.play(race.animate_descent(), run_time=ctx.remaining - 1.0)   # then the three balls roll
```

---

## 7. Domain 2: microelectronics and computer architecture (`viz.microelectronics`)

**Intuition map.** A PN junction is *two crowds mixing until a charged moat stops them*; bias *widens or narrows the moat*.
A MOSFET gate is *a plate that pulls electrons into a bridge under the oxide*, and the drain voltage *tilts the bridge
until it pinches off*. A pipeline is *a laundry line with five stations*; a hazard is *a customer who needs the dryer
before the washer has finished*. A cache is *a short shelf in front of a distant warehouse*.

### 7.1 Pure math (`microelectronics/_math.py`)

| Function | Returns |
|----------|---------|
| `depletion_width(v_bias, na, nd, normalise=True)` | normalised depletion width and built-in potential (abrupt-junction scaling, `~ sqrt(Vbi - V)`) |
| `diode_current(v, i_s, n, vt)` | Shockley diode current |
| `mos_region(vgs, vds, vth)` | `"cutoff"`, `"triode"` or `"saturation"` |
| `drain_current(vgs, vds, vth, k)` | square-law `I_D` (triode and saturation); `channel_profile(vgs, vds, vth)` gives inversion-layer depth along the channel, tapering to pinch-off |
| `pipeline_schedule(instrs, forwarding, branch_policy)` | cycle-by-cycle stage occupancy, stall bubbles with the cause (`RAW`, `load-use`, `control`), and forwarding edges |
| `two_bit_predictor(history)` | state sequence of a 2-bit saturating counter and prediction hits |
| `cache_simulate(addresses, sets, ways, block, policy)` | per access: tag, index, offset, hit or miss, evicted line, LRU order |
| `amat(hit_time, miss_rate, miss_penalty)` | average memory access time (supports multi-level) |
| `dma_schedule(words, cpu_busy)` | bus ownership timeline for CPU and DMA |
| `spi_frame(data, cpol, cpha, bits)` / `i2c_frame(addr, data, rw)` / `uart_frame(byte, parity, stop, bits)` | edge-accurate bit timelines with sampling instants |
| `nrzi_encode(bits, stuffing=True)` / `can_frame(id, data, stuffing=True)` | USB-style NRZI with bit stuffing; CAN frame bits with stuffing and arbitration order |
| `differential(bits, noise, seed)` | `D+`, `D-`, differential and common-mode components |

### 7.2 Semiconductor physics

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `PNJunction` | M | `PNJunction(bias=0.0, n_carriers=24, show_ions=True, seed=7)` | P and N regions, holes (open circles) and electrons (filled dots) as `ParticleField`s, fixed ionised dopants in the depletion region, and the built-in field arrow. `.animate_diffusion()`, `.animate_recombination()` (an electron and hole meet, flash and vanish), `.animate_bias(v_from, v_to)` (**the depletion region widens under reverse bias and shrinks under forward bias**, driven by `depletion_width`). |
| `PNBandDiagram` | M | `PNBandDiagram(junction)` | Energy bands bending across the junction, linked to the same bias tracker. |
| `DiodeIVCurve` | M | `DiodeIVCurve(i_s, n, v_range)` | The I-V curve with an operating point that follows the junction's bias. |
| `MOSFETCrossSection` | M | `MOSFETCrossSection(kind="nmos", vgs=0.0, vds=0.0, vth=1.0)` | Gate metal, oxide, body, n+ source and drain. `.animate_vgs(v_from, v_to)` pulls electrons under the oxide: the **inversion layer** fades in as `vgs` crosses `vth`, with the depletion region beneath it. `.animate_vds(...)` tilts and thins the channel toward the drain until **pinch-off** in saturation. |
| `DrainCurrentVectors` | M | `DrainCurrentVectors(device, scale=1.0)` | Velocity arrows on carriers in the channel; length follows the computed `I_D`, direction source to drain. |
| `MOSFETCharacteristics` | M | `MOSFETCharacteristics(vth, k, vgs_list=(1.5, 2.0, 2.5, 3.0))` | `I_D`-`V_DS` family with the linear/saturation boundary, and the operating point linked to the cross-section. |
| `TransferCurve` | M | `TransferCurve(vth, k)` | `I_D` against `V_GS` with the threshold marked. |

**Scenes:** `play_pn_junction` (`view`: carriers, bands, iv), `play_mosfet_channel`.

### 7.3 CPU pipeline and caches

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `PipelineDiagram` | M | `PipelineDiagram(program, stages=("IF","ID","EX","MEM","WB"), forwarding=True, branch_policy="stall")` | Instruction-by-cycle grid. `.animate_cycle(n)` advances the clock; `.animate_stall(instr, cycles)` inserts **bubbles** (warning colour) labelled with the cause; `.animate_forward(src, dst)` draws a forwarding arrow that removes a stall. |
| `HazardTimeline` | M | `HazardTimeline(program)` | The same program with and without forwarding, stacked, with cycle totals compared. |
| `BranchPredictionPath` | M | `BranchPredictionPath(program, predictor="2bit", outcomes=(1,1,0,1))` | A fork in the instruction stream: predicted path in blue, the squashed wrong path flushed in red; shows the cost of a mispredict in cycles. |
| `PredictorFSM` | M | `PredictorFSM(bits=2)` | The saturating-counter state diagram with the current state lit as the outcomes arrive. |
| `CPUDatapath` | M | `CPUDatapath(program, show_control=False)` | A single-cycle-style datapath (PC, instruction memory, register file, ALU, data memory) with an instruction token travelling through; useful as the zoomed-in view of one pipeline stage. |
| `CacheArray` | M | `CacheArray(sets=4, ways=2, block=4, policy="lru")` | Sets and ways with valid bit, tag and data, and LRU order. `.animate_access(address)`: the address splits into **tag / index / offset**, the set is searched; **hit** flashes green, **miss** flashes red, the block is fetched from the next level and a victim is evicted. |
| `CacheHierarchy` | M | `CacheHierarchy(levels=("L1","L2","L3"), ram=True, latencies=(1, 10, 40, 200))` | Stacked blocks with latency labels and a live AMAT readout. `.animate_lookup(addr, hits=(False, False, True))` sends the request down until it hits and returns the data up. |
| `AddressBus` / `DataBus` | M | `AddressBus(width=8)` | Parallel lines carrying value pulses; `.animate_transfer(value)`. |
| `RAMArray` | M | `RAMArray(rows, cols, highlight=None)` | Row/column decoded memory array with the addressed cell lit. |
| `DMAController` | M | `DMAController(words=8)` | CPU, memory, peripheral and DMA engine; `.animate_transfer()` moves a block while the CPU is shown free to compute (bus ownership timeline underneath). |

**Scenes:** `play_cpu_pipeline`, `play_branch_prediction`, `play_cache_access`, `play_memory_system` (`view`: hierarchy, dma).

### 7.4 Bus protocols

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `DigitalWaveform` | M | `DigitalWaveform(bits, name, bit_time=1.0, style="nrz")` | Generic labelled digital trace with bit boundaries; the base of every bus below. |
| `ClockLine` | M | `ClockLine(cycles, cpol=0)` | Clock with edge markers (`.mark_sampling_edges(edge="rising")`). |
| `SPIBus` | M | `SPIBus(mosi, miso=None, cpol=0, cpha=0)` | `SCLK`, `MOSI`, `MISO`, `CS` traces, the sampling edge marked for the chosen mode, and a shift-register view. |
| `I2CBus` | M | `I2CBus(address, data, read=False)` | `SDA`/`SCL` with START, address, R/W, ACK, data and STOP framed and labelled. |
| `UARTFrame` | M | `UARTFrame(byte, parity="none", stop=1)` | Idle, start bit, LSB-first data, parity, stop; mid-bit sampling ticks. |
| `DifferentialPair` | M | `DifferentialPair(bits, noise=0.0, seed=7)` | `D+` and `D-` traces plus the derived differential and common-mode signals; `.animate_noise()` injects noise equally on both lines and shows the receiver subtracting it away. |
| `USBLine` | M | `USBLine(bits, stuffing=True)` | NRZI encoding on `D+`/`D-` with stuffed bits flagged. |
| `CANBus` | M | `CANBus(frame_bits)` and `CANArbitration(nodes)` | `CAN_H`/`CAN_L` with dominant and recessive levels; arbitration where the lower ID wins bit by bit. |

**Scene:** `play_serial_bus` (`variant`: spi, i2c, uart, usb, can).

### 7.5 Usage

```python
from manimate.viz.microelectronics import PNJunction, DiodeIVCurve
from manimate.viz.core import Slider
from manim import ValueTracker

bias = ValueTracker(0.0)
junction = PNJunction(bias=0.0).bind(bias, "bias")
iv = DiodeIVCurve(i_s=1e-12, n=1.0, v_range=(-1.0, 0.8)).bind(bias, "v")
ctx.play(junction.animate_recombination(), run_time=2.0)
ctx.play(bias.animate.set_value(0.6), run_time=3.0)   # moat narrows; operating point climbs the I-V curve
ctx.play(bias.animate.set_value(-0.8), run_time=3.0)  # moat widens; current stays near zero
```

---

## 8. Domain 3: control, signals, RF and DSP (`viz.signals`)

**Intuition map.** A transfer function is *a landscape over the complex plane whose peaks are poles*; the frequency
response is *the profile you see when you slice that landscape along the imaginary axis*. A Fourier transform is *wrapping
a signal around a circle at each frequency and watching where its centre of mass lands*. A Smith chart is *the reflection
coefficient's unit disc, redrawn so impedance can be read off it*. A pole near the unit circle is *a hill that the
frequency point climbs as it walks around the circle*.

### 8.1 Pure math (`signals/_math.py`)

| Function | Returns |
|----------|---------|
| `tf_from_pz(poles, zeros, gain)` / `tf_eval(tf, s)` | transfer function as arrays; evaluation at complex `s` or `z` |
| `bode_data(tf, w)` | magnitude (dB), phase (deg), gain and phase margins, asymptotic (straight-line) approximation |
| `nyquist_data(tf, w)` | the contour, mirrored branch, and encirclement count about `-1` |
| `root_locus_data(num, den, k_values)` | closed-loop pole branches as gain increases |
| `step_response(tf_or_ss, t)` / `impulse_response(...)` | response with `rise_time`, `overshoot`, `settling_time` |
| `pid_simulate(kp, ki, kd, plant, setpoint, t)` | output and the separate P, I, D contributions |
| `state_space(A, B, C, D)` / `simulate_ss(ss, u, t)` | states and output |
| `waveform(kind, f, amp, phase, t)` | sine, square, sawtooth, triangle; `fourier_coefficients(kind, n_terms)` |
| `winding_centroid(signal, t, f)` | centre of mass of the signal wrapped at frequency `f` (the winding machine) |
| `laplace_magnitude(poles, zeros, gain, sigma, omega)` | `|H(s)|` on a grid for the surface |
| `dipole_pattern(theta, kind)` / `dipole_fields(r, theta, t, k)` | normalised radiation pattern, half-power beamwidth, and E/H amplitudes |
| `reflection_coefficient(zl, z0)` / `vswr(gamma)` / `input_impedance(zl, z0, length_wl)` | `Gamma`, SWR and the impedance seen along a line |
| `standing_wave(gamma, x, t)` | instantaneous voltage and the `|V(x)|` envelope |
| `smith_map(z, z0)` | impedance to Smith-chart coordinates; `r_circle`, `x_arc` for the grid |
| `fir_response(taps, w)` / `iir_response(b, a, w)` / `filter_signal(b, a, x)` | frequency response and filtered output |
| `fft_butterfly_graph(n)` | stage-by-stage butterfly pairs with twiddle factors and bit-reversed order |
| `alias_frequency(f, fs)` | the apparent frequency after sampling |

### 8.2 Block diagrams and control

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `TransferFunctionBlock` | M | `TransferFunctionBlock(numerator=None, denominator=None, latex=None, poles=None, zeros=None)` | A box showing `H(s)`, with an optional pole-zero inset. |
| `SummingJunction` | M | `SummingJunction(signs=("+","-"), radius=0.25)` | Circle with signed inputs. |
| `FeedbackLoop` | M | `FeedbackLoop(forward=[...], feedback=[...], reference="r", output="y")` | Forward and feedback paths with a take-off point. `.animate_signal(input_fn)` sends pulses around the loop. |
| `PIDController` | M | `PIDController(kp, ki, kd)` | Three parallel paths (P, I, D) summing to `u`; `.animate_response(error_fn)` shows each path's contribution as a small scope. |
| `StateSpaceDiagram` | M | `StateSpaceDiagram(A, B, C, D)` | Integrator chain with `A`, `B`, `C`, `D` blocks drawn as gains and feedback. |
| `BlockDiagram` | M | `BlockDiagram(nodes, edges, layout="layered")` | General builder from a dict spec with automatic left-to-right layout; `.tag_scope(node)` attaches a live `ScrollingScope`. |
| `StepResponsePlot` | M | `StepResponsePlot(tf, t_end, kind="step", annotate=True)` | The step (or, with `kind="impulse"`, impulse) response drawing itself, then **rise time**, **overshoot** and **settling band** annotations. |
| `PIDTuningDemo` | M | `PIDTuningDemo(plant, kp, ki, kd, setpoint=1.0)` | Output with the three terms; sliders drive `kp`, `ki`, `kd` so the narrator can show P-only offset, then I removing it, then D damping. |
| `PoleZeroPlot` | M | `PoleZeroPlot(poles, zeros, domain="s")` | `s`-plane (with stable half shaded) or `z`-plane (with unit circle). `.animate_move_pole(idx, to)`. |
| `BodePlot` | M | `BodePlot(poles, zeros, gain=1.0, w_range=(0.01, 100), asymptotes=True, margins=True)` | Magnitude (dB) over phase, sharing a log axis, with straight-line asymptotes and the gain and phase margins marked. |
| `NyquistPlot` | M | `NyquistPlot(poles, zeros, gain=1.0)` | The contour in the complex plane, the critical point `-1`, and the encirclement count. `.animate_trace()`. |
| `RootLocus` | M | `RootLocus(num, den, k_max=10)` | Branches drawn as gain sweeps; `.animate_k(k)` shows the closed-loop poles moving. |

### 8.3 Waveforms, phasors and transforms

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `WaveGenerator` | M | `WaveGenerator(kind="sine", freq=1.0, amp=1.0, phase=0.0)` | A wave with `Slider`s for frequency, amplitude and phase; kinds: sine, cosine, square, sawtooth, triangle. Driven by `ValueTracker`s, so the trace morphs live. |
| `PhasorWheel` | M | `PhasorWheel(phasors, show_projection=True)` | Rotating arrows on the complex plane with a side trace of their projection (the wheel *is* the wave). `.animate_rotate(turns)`. |
| `PhasorSum` | M | `PhasorSum(phasors)` | Phasors added tip to tail, with the resultant and its waveform. |
| `FourierSeriesBuilder` | M | `FourierSeriesBuilder(kind="square", n_terms=5)` | Stacked rotating circles whose tip draws the waveform; `.animate_add_term()` shows the approximation sharpening, including overshoot at the edges. |
| `WindingFourier` | M | `WindingFourier(signal_fn, f_tracker, t_range=(0, 4))` | The winding machine: the signal wrapped around a circle at frequency `f`, its centre of mass plotted against `f` (a spectrum emerges as `f` sweeps). |
| `SpectrumMorph` | M | `SpectrumMorph(signal_fn, t_range)` | The time-domain trace morphing into spectrum bars (Fourier **time-to-frequency** transition). |
| `LaplaceSurface` | M | `LaplaceSurface(poles, zeros, gain=1.0, projector=None, sigma_range=(-3, 1), omega_range=(-4, 4))` | A 2.5D `|H(s)|` surface over the `s`-plane: **poles are peaks, zeros are pits**, and the `j-omega` slice is highlighted and lifted off into a Bode-style curve (the **Laplace-to-frequency** transition). |
| `Spectrogram` | M | `Spectrogram(signal_fn, window=0.5, hop=0.1)` | Time-frequency heatmap for chirps and notes. |

**Scenes:** `play_block_diagram`, `play_pid_response`, `play_bode_plot`, `play_nyquist_plot`, `play_root_locus`,
`play_time_response` (`kind`: step, impulse), `play_fourier_transform` (`view`: winding, spectrum, series), `play_waveform_generator`. The `LaplaceSurface` picture is the `surface` view of `play_pole_zero_map` (section 8.5).

### 8.4 RF and electromagnetics

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `DipoleAntenna` | M | `DipoleAntenna(length_wl=0.5, orientation=UP)` | Two arms with a feed and the current distribution along them. |
| `RadiationPattern2D` | M | `RadiationPattern2D(kind="half_wave", plane="E")` | Polar plot with lobes, nulls and a half-power beamwidth wedge. |
| `RadiationPattern3D` | M | `RadiationPattern3D(kind="half_wave", projector=None)` | The doughnut as a surface of revolution via `Surface2p5D`; `.animate_rotate()`. |
| `EMWave` | M | `EMWave(wavelength=2.0, direction="z", e_axis="x", amplitude=1.0)` | **E** (blue) and **H** (yellow) as orthogonal sinusoidal vector fields propagating along the Poynting vector (teal arrow), in a 2.5D view. `.animate_propagate()`. |
| `FieldRadiation` | M | `FieldRadiation(freq, frames=12)` | Closed field loops pinching off the dipole and travelling outward (near field to far field). |
| `TransmissionLine` | M | `TransmissionLine(z0=50, zl=(25, 25), length_wl=1.0)` | Source, line and load. `.animate_pulse()` launches a pulse; `ReflectionPulse` bounces it with amplitude `Gamma`; a lattice (bounce) diagram shows the multiple reflections. |
| `StandingWaveEnvelope` | M | `StandingWaveEnvelope(gamma, wavelength=1.0)` | Instantaneous wave inside its `|V(x)|` envelope (via `AreaBand`), with `V_max`, `V_min` and a live **SWR** readout. |
| `SmithChart` | M | `SmithChart(z0=50, grid="rx", show_swr_circle=False)` | Constant-resistance circles and reactance arcs, clean and uncluttered. |
| `ImpedanceLocus` | M | `ImpedanceLocus(chart, zl, length_wl=0.5)` | The load point rotating **clockwise toward the generator** around the SWR circle as the line gets longer, with the `Gamma` phasor from the centre. `.animate_toward_generator(length_wl)`. |

**Scenes:** `play_antenna_pattern` (`dims`: 2d, 3d; `show_fields`), `play_transmission_line` (`view`: pulse, standing_wave), `play_smith_chart`.

### 8.5 Digital signal processing

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `TapDelayLine` | M | `TapDelayLine(coeffs, labels=None)` | FIR structure: `z^-1` delay boxes, multipliers and an adder. `.animate_sample(x)` shifts samples along the line and builds the output `y[n]`. |
| `IIRBiquad` | M | `IIRBiquad(b, a, form="direct2")` | Direct-form structure with feedback paths drawn distinctly. |
| `PoleZeroToResponse` | M | `PoleZeroToResponse(poles, zeros)` | `z`-plane beside the response curve. A point walks the unit circle; vectors from each zero and pole to it show **gain = product of zero distances over product of pole distances**, so a pole near the circle makes a peak. |
| `FFTButterfly` | M | `FFTButterfly(n=8)` | Butterfly diagram with twiddle factors; `.animate_stage(s)` highlights the pairs combined in stage `s`; `.show_bit_reversal()`. |
| `SamplingDemo` | M | `SamplingDemo(f_signal, fs)` | Samples on a wave and the alias it implies; sweeping `f_signal` past `fs/2` makes the apparent frequency fold back. |

**Scenes:** `play_digital_filter` (`kind`: fir, iir), `play_fft_butterfly`, `play_sampling_aliasing`, `play_pole_zero_map` (`domain`: s, z; `view`: plane, surface, response).

### 8.6 Usage

```python
from manimate.viz.signals import LaplaceSurface, BodePlot
from manimate.viz.core import Projector3D

surf = LaplaceSurface(poles=[-0.5 + 2j, -0.5 - 2j], zeros=[], projector=Projector3D(yaw=-55, pitch=30))
bode = BodePlot(poles=[-0.5 + 2j, -0.5 - 2j], zeros=[], gain=4.0).scale(0.7).to_edge(RIGHT)
fade_in_whole(ctx, surf)                      # two peaks over the s-plane
ctx.play(surf.animate_slice_jw(), run_time=2)  # slice along the imaginary axis...
ctx.play(Transform(surf.slice_curve, bode.magnitude_curve), run_time=2)  # ...and it becomes the Bode magnitude
```

---

## 9. Domain 4: power systems, power electronics and electrical machines (`viz.power`)

**Intuition map.** A converter is *a valve that chops energy into packets and a reservoir that smooths them*; the
inductor current is *a ramp up while the switch is on and a ramp down while it is off*. PWM is *comparing a slow wave to a fast
ramp*. The `d-q` frame is *a camera riding on the rotor, from which the spinning currents look still*. Rotor angle `delta` is
*a ball oscillating in the dip of a sinusoidal power-angle bowl*.

> Reuse note: the repo already ships `ac_waveform`, `phasor_diagram`, `rotating_field`, `synchronous_motor`,
> `transformer` and `circuit_diagram`. `viz.power` does not replace them. It adds the converter, drive and transient
> pictures that are missing and, over time, shares their drawing helpers.

### 9.1 Pure math (`power/_math.py`)

| Function | Returns |
|----------|---------|
| `converter_waveforms(topology, vin, duty, L, C, R, fsw, periods)` | switch-node voltage, inductor current (with `di/dt` per interval), output voltage, in CCM and DCM, for `buck`, `boost`, `buck_boost` |
| `conversion_ratio(topology, duty)` | `Vout/Vin` |
| `hbridge_states(modulation, ma, fsw_ratio)` | switch states (Q1 to Q4), output voltage and load current direction for unipolar and bipolar PWM |
| `pwm_compare(reference, carrier)` | gate pulses from a comparator |
| `spwm_three_phase(vdc, ma, fsw_ratio, periods)` | three pole-voltage pulse trains, line-to-line voltage and the filtered fundamental |
| `clarke(abc)` / `park(ab, theta)` / `inv_park(dq, theta)` | coordinate transforms (power-invariant form stated in the docstring) |
| `foc_currents(id, iq, theta)` | stator currents, flux and torque (`T ~ iq`) for a surface PM machine |
| `rmf_vectors(t, f, phases=3)` | three pulsating vectors and their rotating resultant |
| `swing_simulate(H, Pm, Pmax, delta0, D, disturbance, t)` | rotor angle and speed deviation (classical model) |
| `power_angle_curve(Pmax, delta)` / `equal_area(Pm, Pmax_pre, Pmax_fault, Pmax_post, delta_clear)` | the curve, the areas `A_acc` and `A_dec`, critical clearing angle |
| `fault_current(x_d2, x_d1, x_d, t_d2, t_d1, t_a, phase_angle)` | symmetrical decay plus DC offset: the three-region transient envelope |
| `damper_torque(slip, ...)` | induced bar current and braking torque versus slip |

### 9.2 Grid components

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `Bus` | M | `Bus(name, kv=None, length=2.0)` | A substation bus bar (thick line) with ports. |
| `Breaker` | M | `Breaker(closed=True)` | Breaker symbol; `.animate_trip()`. |
| `TransmissionLineSymbol` | M | `TransmissionLineSymbol(length_km, kv, model="lumped")` | A line; `model="pi"` expands into the `R`, `X`, `B/2` pi circuit. |
| `TransformerSymbol` | M | `TransformerSymbol(ratio, connection="Dyn11")` | Two windings with the shared core; `.animate_flux()` shows the alternating flux. |
| `SynchronousGeneratorSymbol` | M | `SynchronousGeneratorSymbol(rating_mva, poles=2)` | Circle symbol with a small rotating rotor and field. |
| `Substation` | M | `Substation(scheme="double", feeders=3)` | Bus scheme (`single`, `double`, `ring`) with breakers and disconnectors; `.animate_switching(path)` reroutes the current. |
| `PowerGrid` | M | `PowerGrid(nodes, edges, flows=None)` | A one-line diagram from a dict spec; `PowerFlowArrows` scale with `P` and `Q`; `.animate_fault(at)` flashes the fault and sends a current surge from the sources, `.animate_trip(breakers)` isolates it. |

**Scene:** `play_power_grid` (`layout`: radial, ring, substation).

### 9.3 Power electronics converters

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `ConverterTopology` | M | `ConverterTopology(kind="buck", vin=12.0, duty=0.5, L=100e-6, C=100e-6, R=5.0, fsw=100e3)` | Schematic for `buck`, `boost` or `buck_boost`. The **active current path** is highlighted (one colour for switch ON, another for OFF) and the switch symbol flips; `.animate_switching(periods)`. |
| `InductorRamp` | M | `InductorRamp(converter)` | Scopes for switch-node voltage, **inductor current ramp** (slope `di/dt = (Vin - Vout)/L` while the switch is on and `-Vout/L` while it is off, both annotated) and output ripple; bound to the converter's duty tracker. |
| `ConversionRatioCurve` | M | `ConversionRatioCurve(kind)` | `Vout/Vin` against duty cycle with the operating point following the slider. |
| `HBridge` | M | `HBridge(modulation="bipolar", ma=0.8, load="RL")` | Four switches Q1 to Q4 with the diagonal pairs lit; the load current direction flips with the active pair. |
| `PWMGenerator` | M | `PWMGenerator(reference="sine", carrier="triangle", ma=0.8, ratio=15)` | The comparator picture: a slow reference against a fast carrier and the resulting pulse train underneath, with crossings marked. |
| `ThreePhaseInverter` | M | `ThreePhaseInverter(vdc=400, ma=0.9, ratio=21)` | Six-switch bridge with three reference waves 120 degrees apart against one carrier; the three PWM pole voltages build up row by row. |
| `PWMToSine` | U | `PWMToSine(inverter, filter_hz)` | A moving-average window slides over the pulses and **the filtered output settles into a clean three-phase sine**: DC bus in, AC out. |

**Scenes:** `play_dc_dc_converter`, `play_h_bridge`, `play_three_phase_inverter`, `play_pwm_comparator`.

### 9.4 Machines and drives

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `Stator3Phase` | M | `Stator3Phase(poles=2, slots=12)` | Slots with three coloured phase windings (reuses `PHASE_COLORS`). |
| `RotatingMagneticField` | M | `RotatingMagneticField(freq=50, poles=2, show_phases=True)` | Three pulsating phase vectors with their **rotating resultant**; `.animate_rotate(turns)`. |
| `RotorFrame` | M | `RotorFrame(kind="pm")` | Rotor with flux axis; kinds `pm`, `salient`, `induction`. |
| `DQFrame` | M | `DQFrame(theta_tracker, show_abc=True, show_alpha_beta=True)` | Stationary `abc` and `alpha-beta` axes plus the `d-q` axes **rotating in sync with the rotor flux vector** (`d` always on the flux). |
| `ClarkeParkTransform` | M | `ClarkeParkTransform(freq=50, amp=1.0)` | Three sinusoids, then two (`alpha-beta`), then two **constants** (`d-q`), morphing step by step with scopes. |
| `TorqueDecomposition` | M | `TorqueDecomposition(id, iq)` | The stator current vector split into `i_d` (flux) and `i_q` (torque) components with a torque readout `~ i_q`. |
| `FOCBlockDiagram` | M | `FOCBlockDiagram()` | The full loop (speed PI, current PIs, inverse Park, modulator, inverter, motor, Clarke/Park feedback, encoder), built on `BlockDiagram`; `.animate_signal()` runs a command through it. |
| `DamperWindings` | M | `DamperWindings(poles=4, slip=0.05)` | Salient-pole rotor cross-section with damper (amortisseur) bars; at nonzero slip, **induced bar currents** (arrows) appear and a braking torque arrow opposes the relative motion; at synchronous speed they vanish. |
| `RotorAngleOscillation` | M | `RotorAngleOscillation(H=5, Pm=0.8, Pmax=1.8, D=0.05, disturbance=0.5)` | The power-angle curve `P = Pmax sin(delta)` with the operating point oscillating about equilibrium, and a linked `delta(t)` trace. |
| `EqualAreaCriterion` | M | `EqualAreaCriterion(Pm, Pmax_pre, Pmax_fault, Pmax_post, delta_clear)` | Accelerating area `A_acc` and decelerating area `A_dec` shaded and growing; stability is "the second area catches up with the first". |
| `FaultCurrentTransient` | M | `FaultCurrentTransient(x_d2, x_d1, x_d, dc_offset=True)` | Short-circuit current: the envelope decays through **subtransient, transient and steady-state** regions (annotated) with the decaying DC offset making the waveform asymmetric. |

**Scenes:** `play_foc_transform` (`view`: clarke, park, dq_frame, block_diagram), `play_damper_windings`,
`play_swing_rotor_angle` (`view`: power_angle, time), `play_equal_area_criterion`, `play_fault_current`.

### 9.5 Usage

```python
from manimate.viz.power import ConverterTopology, InductorRamp
from manim import ValueTracker

duty = ValueTracker(0.4)
buck = ConverterTopology(kind="buck", vin=12, duty=0.4).bind(duty, "duty")
ramp = InductorRamp(buck).next_to(buck, RIGHT)
fade_in_whole(ctx, VGroup(buck, ramp))
ctx.play(buck.animate_switching(periods=3), run_time=3)       # the active loop lights, current ramps up and down
ctx.play(duty.animate.set_value(0.7), run_time=3)             # Vout rises with duty; the ramp's on-time stretches
```

---

## 10. Domain 5: quantitative finance (`viz.finance`)

**Intuition map.** An option's value is *a smoothed hockey stick*; delta is *the slope of that curve*, gamma is *how
fast the slope changes (its curvature)*, and theta is *the curve sagging toward the payoff as time runs out*. Volatility
clustering is *a band that breathes*. An order book is *two queues facing each other across a spread*. A bond's price
against yield is *a convex slide*; duration is *its tangent*. A binomial tree is *a lattice of possible futures priced by
working backward from the end*.

> Educational visuals only. Nothing here is investment advice, and each scene states its model assumptions
> (risk-neutral pricing, constant volatility, and so on) on screen.

### 10.1 Pure math (`finance/_math.py`)

| Function | Returns |
|----------|---------|
| `norm_cdf(x)` / `norm_pdf(x)` | via `math.erf` (no SciPy needed) |
| `bs_price(S, K, tau, r, sigma, kind)` | Black-Scholes price |
| `bs_greeks(S, K, tau, r, sigma, kind)` | `delta`, `gamma`, `theta`, `vega`, `rho` (vectorised over `S` and `tau` for surfaces) |
| `payoff(kind, S, K, premium)` / `strategy_payoff(legs, S)` | single option and spread, straddle, strangle, collar payoffs |
| `gbm_paths(S0, mu, sigma, T, steps, n, seed)` / `random_walk_paths(steps, n, seed)` | simulated paths |
| `barrier_outcome(paths, barrier, kind, K)` / `asian_average(path)` | knock-in/out flags, payoffs and running averages |
| `garch11(omega, alpha, beta, n, seed, shock_at=None)` | returns and conditional variance series; `unconditional_variance(...)` |
| `implied_vol_surface(strikes, maturities, level, skew, term_slope)` | a smooth, arbitrage-reasonable demo surface (parametric smile plus term structure) |
| `match_order(book, order)` | trades produced under **price-time priority**, the remaining book and partial fills |
| `bond_price(face, coupon, y, n, freq)` | price; `macaulay_duration`, `modified_duration`, `convexity`, `price_approx(y0, dy)` (first and second order) |
| `nelson_siegel(tau, level, slope, curvature, lam)` | a smooth yield curve; `shift_curve(curve, kind, size)` for parallel, steepener, flattener, inversion |
| `crr_tree(S0, u, d, r, dt, n)` / `tree_price(tree, K, kind, american=False)` | recombining lattice, risk-neutral `p`, backward-induction values |

### 10.2 Derivatives and exotic options

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `OptionPayoff` | M | `OptionPayoff(kind="call", strike=100, premium=0, side="long")` and `OptionStrategy(legs)` | The hockey stick with break-even and max loss/gain marked; strategies add legs one at a time. |
| `OptionValueCurve` | M | `OptionValueCurve(kind, strike, sigma, r, tau_tracker)` | Value against spot, with a **tangent line (delta)** and a **curvature bowl (gamma)** at a movable spot; as `tau` shrinks the curve sharpens toward the payoff (theta decay). |
| `GreeksSurface` | M | `GreeksSurface(greek="delta", option="call", strike=100, r=0.03, sigma=0.25, projector=None)` | A 2.5D surface of one Greek against spot and time to expiry. `.animate_time_decay()` slides a slice along the time axis while a 2D profile updates beside it; `greek` is one of `delta`, `gamma`, `theta`, `vega`, `rho`. |
| `GreeksPanel` | M | `GreeksPanel(strike, sigma, r, tau_tracker)` | Small multiples of Delta, Gamma and Theta against spot, all driven by one time slider. |
| `BarrierOptionPaths` | M | `BarrierOptionPaths(paths, barrier, kind="up_and_out", strike)` | Paths drawn together; those that cross the barrier are knocked out (turn red and fade) and the survivors' payoffs are summed. |
| `AsianOptionPath` | M | `AsianOptionPath(path, strike)` | A spot path with its running average; the payoff uses the average, not the final price. |

**Scenes:** `play_option_payoff`, `play_option_greeks` (`view`: surface, curves), `play_exotic_option_paths`.

### 10.3 Volatility

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `GARCHSeries` | M | `GARCHSeries(returns, variance, band_k=2.0)` | Returns as bars over a **variance band** (`+/- k sigma_t`) that expands after large moves and contracts in calm periods (via `AreaBand`). |
| `GARCHRecursion` | M | `GARCHRecursion(omega, alpha, beta)` | The recursion `sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2` with the three terms as coloured blocks sized by contribution. |
| `VolatilityShock` | M | `VolatilityShock(alpha, beta, shock_at)` | A single shock and the geometric decay of variance back to its long-run level (persistence = `alpha + beta`). |
| `VolSurfaceWireframe` | M | `VolSurfaceWireframe(strikes, maturities, iv, projector=None)` | The implied-volatility surface as a wireframe, with a highlighted smile slice and term-structure slice. |

**Scenes:** `play_garch_volatility`, `play_vol_surface`.

### 10.4 Market microstructure and fixed income

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `OrderBookStack` | M | `OrderBookStack(bids, asks, depth=5)` | Bid and ask ladders with bar lengths proportional to size, and the **spread** highlighted. |
| `MatchingEngine` | M | `MatchingEngine(book, orders)` | `.animate_order(side, price, qty, kind="limit")`: a marketable order sweeps price levels, each fill flashes, partial fills leave a remainder, and an unmarketable limit order joins its queue (price-time priority shown as queue order). |
| `TradeTape` | M | `TradeTape(trades, max_rows=6)` | Scrolling prints of `price x size`, with colour for aggressor side. |
| `DepthChart` | M | `DepthChart(bids, asks)` | Cumulative step depth either side of the mid-price. |
| `BondPriceYieldCurve` | M | `BondPriceYieldCurve(face=100, coupon=0.05, maturity=10, y_range=(0.0, 0.12))` | The inverse, convex price-yield relationship with a moving point. `.show_duration(y0)` draws the tangent; `.show_convexity(y0, dy)` shows first-order (line) vs second-order (parabola) vs actual. |
| `CashflowLadder` | M | `CashflowLadder(bond, y)` | Coupon and principal bars shrinking into their present values (discounting made visible). |
| `YieldCurve` | M | `YieldCurve(maturities, yields=None, ns_params=None)` | The curve with `.animate_shift(kind, size)` for **parallel shift, steepening, flattening and inversion**, labelled with what each says about expectations. |

**Scenes:** `play_order_book_matching`, `play_bond_price_yield` (`view`: price_yield, duration_convexity), `play_yield_curve_shift`.

### 10.5 Stochastic processes

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `BrownianPaths` | M | `BrownianPaths(n_paths=12, T=1.0, steps=200, drift=0.0, vol=1.0, seed=7)` | Paths drawn together with a histogram of terminal values forming on the right (the bell curve emerges). |
| `RandomWalkToBrownian` | M | `RandomWalkToBrownian(step_counts=(10, 40, 160, 640))` | The same walk with ever finer steps converging to Brownian motion (rescaled so the shape stabilises). |
| `GBMFan` | M | `GBMFan(S0, mu, sigma, T, n_paths=30, percentiles=(5, 50, 95))` | Geometric Brownian paths with a percentile fan (via `AreaBand`). |
| `BinomialTree` | M | `BinomialTree(S0=100, up=1.1, down=0.9, steps=3, r=0.02, strike=None)` | A recombining lattice with node prices; `.animate_backward_induction()` fills option values from the leaves back to the root using the risk-neutral probability `p`. |
| `BinomialConvergence` | M | `BinomialConvergence(S0, K, r, sigma, T, n_max=60)` | Tree price against number of steps converging onto the Black-Scholes value (a dashed line). |

**Scenes:** `play_brownian_motion` (`variant`: random_walk, brownian, gbm), `play_binomial_tree`.

### 10.6 Usage

```python
from manimate.viz.finance import GreeksSurface, OptionValueCurve
from manimate.viz.core import Projector3D

surf = GreeksSurface("gamma", option="call", strike=100, sigma=0.25, projector=Projector3D(yaw=-60, pitch=28))
fade_in_whole(ctx, surf)
ctx.play(surf.animate_time_decay(), run_time=ctx.remaining - 1)   # the gamma ridge sharpens as expiry nears
```

---

## 11. Domain 6: quantum computing and applied math (`viz.quantum_math`)

**Intuition map.** A qubit is *an arrow on a sphere*; a single-qubit gate is *a rotation of that sphere about an axis*;
measurement is *dropping the arrow onto a pole with probability set by how close it is*. Entanglement is *an arrow that
is forced inside the ball*: one qubit alone has a shorter vector than a pure state. A phase portrait is *a flow you can
sit in and watch*; fixed points are *where the flow is still*. An eigenvector is *a direction the space refuses to turn*.

### 11.1 Pure math (`quantum_math/_math.py`)

| Function | Returns |
|----------|---------|
| `GATES` | `I, X, Y, Z, H, S, T` and `rx(theta), ry(theta), rz(theta)` as NumPy matrices |
| `bloch_vector(state)` / `state_from_bloch(theta, phi)` | `(x, y, z)` and back |
| `gate_axis_angle(gate)` | rotation axis and angle on the Bloch sphere (for example `H`: axis `(x+z)/sqrt(2)`, angle `pi`); `rotation_path(gate, n)` gives the arc the vector sweeps |
| `simulate(circuit, n_qubits)` | state vector for up to 5 qubits; `probabilities(state)`; `sample(state, shots, seed)` |
| `reduced_bloch(state, qubit)` | partial trace to the Bloch vector of one qubit (length < 1 when entangled) |
| `eig2(matrix)` | eigenvalues and eigenvectors (real or complex pair), with `classification` |
| `svd2(matrix)` | `U`, singular values, `Vt` |
| `rk4(system, y0, t, dt)` | trajectory integration; `streamline(field, seed_point, steps)` |
| `find_fixed_points(system, bounds)` / `classify_fixed_point(J)` | fixed points with their linearisation and type: `saddle`, `stable node`, `unstable node`, `stable spiral`, `unstable spiral`, `centre` |
| `nullclines(system, bounds)` | the curves `dx/dt = 0` and `dy/dt = 0` |
| `SYSTEMS` | presets: `damped_pendulum`, `lotka_volterra`, `van_der_pol`, `saddle`, `spiral_sink`, `duffing` |

### 11.2 Quantum information

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `BlochSphere` | M | `BlochSphere(radius=2.0, projector=None, labels=True, equator=True)` | A depth-shaded sphere (`Sphere2p5D`) with axes and the six cardinal states `\|0>, \|1>, \|+>, \|->, \|+i>, \|-i>`. |
| `StateVector` | M | `StateVector(sphere, theta=0.0, phi=0.0, trail=True)` | The arrow with a glowing tip, dashed projections onto the axes and a trail. Accepts `(alpha, beta)` too. |
| `StateVector.animate_gate` | U | `.animate_gate(gate, run_time=1.5)` | Rotates the arrow along the true arc about the gate's axis for `X`, `Y`, `Z`, `H`, `S`, `T` or `("Rx", theta)`, drawing the rotation axis and an angle label. `.animate_sequence(gates)` chains them. |
| `.animate_measure` | U | `.animate_measure(basis="Z", outcome=None, seed=7)` | The arrow collapses to a pole; the probabilities appear first, so the viewer sees why this outcome was likely. |
| `MixedBlochBall` | M | `MixedBlochBall(sphere, vector)` | A shorter-than-unit vector inside the ball: the reduced state of one half of an entangled pair. |
| `QuantumCircuit` | M | `QuantumCircuit(n_qubits, gates, labels=None)` | Wires and gates (`H`, `X`, `Z`, `S`, `T`, `Rx`, ... , `CNOT` with control dot and target, `CZ`, `SWAP`, `Measure`, `Barrier`). `.animate_run()` moves a playhead through the circuit and updates the readouts after each column. |
| `AmplitudeBars` | M | `AmplitudeBars(state, basis_labels=None)` | Amplitude magnitudes as bars with **hue = phase**, so interference is visible. |
| `ProbabilityHistogram` | M | `ProbabilityHistogram(probs, labels=None)` | Basis-state probabilities. `.animate_to(new_probs)` morphs on every gate; `.animate_shots(n, seed)` builds the histogram from sampled measurements. |
| `EntanglementDemo` | M | `EntanglementDemo(preset="bell")` | Circuit, histogram and the reduced Bloch vectors together: after the CNOT the individual vectors shrink to the centre while the joint outcomes stay perfectly correlated. Presets `bell`, `ghz`. |

**Scenes:** `play_bloch_sphere`, `play_quantum_circuit` (`preset`: bell, ghz, custom; `shots`).

### 11.3 Differential equations and dynamical systems

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `VectorField2D` | M | `VectorField2D(system, x_range, y_range, density=14, color_by="magnitude")` | Arrows coloured by speed; `system` is a callable or a preset name. |
| `StreamLines` | M | `StreamLines(field, n=18, speed=1.0)` | Dashes flowing along trajectories via an updater, so the field reads as motion. |
| `PhasePortrait` | M | `PhasePortrait(system, ics=None, nullclines=True, fixed_points=True, x_range, y_range)` | Trajectories from several initial conditions, nullclines, and fixed points **detected and labelled** (saddle, sink, source, spiral, centre). |
| `JacobianLens` | M | `JacobianLens(portrait, point)` | Zooms into a fixed point; the nonlinear flow melts into the linear flow with the eigenvectors drawn (the link to the next block). |
| `LimitCycleDemo` | M | `LimitCycleDemo(system="van_der_pol", mu=1.5)` | Trajectories from inside and outside spiralling onto one closed orbit. |

**Scene:** `play_phase_portrait` (`system`: any preset, or `dx` / `dy` as `safe_expr` strings in `x`, `y`).

### 11.4 Linear algebra

| Name | Kind | Signature | Shows / key methods |
|------|------|-----------|---------------------|
| `TransformableGrid` | M | `TransformableGrid(matrix=None, extent=4, show_basis=True)` | A grid and basis vectors; `.animate_apply(matrix)` warps space (matrix multiplication as stretching and shearing). Also shows the determinant as the area of the image of the unit square. |
| `EigenvectorOverlay` | M | `EigenvectorOverlay(grid, matrix)` | Lines through the eigen-directions: as the grid warps, **vectors on those lines stay on them (only scaled by the eigenvalue)** while any other vector turns. Complex eigenvalues are shown as a rotation-and-scale (`ComplexEigenSpiral`). |
| `EigenDecompositionStory` | M | `EigenDecompositionStory(matrix)` | `A = P D P^-1` as three moves: change to the eigenbasis, scale along the axes, change back; the three-step result equals the direct transform. |
| `SVDStory` | M | `SVDStory(matrix)` | The unit circle becomes an ellipse: rotate (`V^T`), stretch by the singular values, rotate (`U`); the ellipse's semi-axes are the singular values. |
| `PowerIterationDemo` | M | `PowerIterationDemo(matrix, v0, steps=8)` | Repeated application turning any vector toward the dominant eigenvector. |

**Scenes:** `play_linear_transformation` (adopts roadmap #17), `play_eigen_decomposition` (`view`: eigen, decomposition, svd).

### 11.5 Usage

```python
from manimate.viz.quantum_math import BlochSphere, StateVector, ProbabilityHistogram
from manimate.viz.core import Projector3D

sphere = BlochSphere(projector=Projector3D(yaw=-35, pitch=22))
psi = StateVector(sphere, theta=0, phi=0)            # starts at |0>
fade_in_whole(ctx, VGroup(sphere, psi))
ctx.play(psi.animate_gate("H"), run_time=1.5)         # rotates about (x+z)/sqrt(2) to |+>
ctx.play(psi.animate_gate(("Rz", 3.14159 / 2)), run_time=1.5)
ctx.play(psi.animate_measure(basis="Z"), run_time=1.5)
```

---

## 12. Registry adapters and the catalog budget

### 12.1 Adapter list

These are the only new entries the planner will ever see. Each is a thin `@component` over a `play_*` scene. Parameters
are deliberately few; `view` / `variant` / `preset` choices select between related pictures. Component names avoid every
existing and reserved name (section 2.3); the four adopted roadmap names are marked **R**.

How adapters express things the registry cannot (all verified against `registry.py`):

- **Complex numbers** (poles, zeros) are passed as flat `[re, im, re, im, ...]` in a `number[]` (at most 8 items, so up to four values).
- **A 2 x 2 matrix** is a row-major `number[]` of length 4. Larger matrices are never taken from the planner; they come from presets or a `seed`.
- **Choice lists.** `choices` is checked against the *whole* value, so list-valued params use a `validator` (each item must be in an allowed set) rather than `choices`. Scalar options use `choices`.
- **Expressions** (`dx`, `dy`, loss functions) are strings compiled by the extended `safe_expr`; there is no other path to arbitrary maths.

| Component | Domain tag | Key params (max 5) | Wraps |
|-----------|-----------|--------------------|-------|
| `attention_head` | machine-learning | `tokens[]`, `query_index`, `causal`, `heads` (1-4) | `AttentionHead`, `MultiHeadAttention` |
| `attention_heatmap` **R** | machine-learning | `tokens[]`, `query_index`, `causal`, `view` (heatmap, arcs, both) | `AttentionHeatmap`, `AttentionArcs` |
| `positional_encoding` | machine-learning | `view` (waves, sum, heatmap), `tokens[]`, `d_model`, `n_positions` | `PositionalEncodingWaves`, `PositionalEncodingSum`, `PEHeatmap` |
| `kv_cache` | machine-learning | `tokens[]`, `window`, `compare_no_cache` | `KVCacheBuffer`, `DecodeTimeline` |
| `diffusion_process` | machine-learning | `variant` (image, points), `view` (forward, reverse, both), `steps`, `schedule` | `ForwardDiffusion`, `ReverseDiffusion`, `Distribution2DDiffusion` |
| `autoencoder_latent` | machine-learning | `variant` (ae, vae), `layers[]`, `interpolate` | `Autoencoder`, `LatentSpace2D`, `LatentInterpolation`, `VAESampling` |
| `gradient_descent` **R** | machine-learning | `landscape` (bowl, ravine, saddle, two_minima, beale), `optimizer` (sgd, momentum, adam, race), `learning_rate`, `start[]` | `LossLandscape`, `DescentPath`, `OptimizerRace` |
| `backprop_flow` | machine-learning | `layers[]`, `activation`, `show_chain_rule` | `BackpropNetwork`, `ChainRuleStrip` |
| `pn_junction` | microelectronics | `bias_v`, `view` (carriers, bands, iv) | `PNJunction`, `PNBandDiagram`, `DiodeIVCurve` |
| `mosfet_channel` | microelectronics | `kind` (nmos, pmos), `vgs`, `vds`, `view` (cross_section, characteristics) | `MOSFETCrossSection`, `DrainCurrentVectors`, `MOSFETCharacteristics` |
| `cpu_pipeline` | microelectronics | `program` (raw_hazard, load_use, independent, branch), `forwarding`, `branch_policy` | `PipelineDiagram`, `HazardTimeline` |
| `branch_prediction` | microelectronics | `predictor` (static, 1bit, 2bit), `outcomes[]` | `BranchPredictionPath`, `PredictorFSM` |
| `cache_access` | microelectronics | `addresses[]`, `sets`, `ways`, `policy` (lru, fifo) | `CacheArray` |
| `memory_system` | microelectronics | `view` (hierarchy, dma), `levels`, `words` | `CacheHierarchy`, `AddressBus`, `RAMArray`, `DMAController` |
| `serial_bus` | microelectronics | `variant` (spi, i2c, uart, usb, can), `data[]`, `mode` | `SPIBus`, `I2CBus`, `UARTFrame`, `USBLine`, `CANBus`, `DifferentialPair` |
| `block_diagram` | signals | `preset` (feedback, pid, state_space, cascade), `input` (step, sine, ramp), `show_scopes` | `FeedbackLoop`, `PIDController`, `StateSpaceDiagram`, `BlockDiagram` |
| `pid_response` | signals | `plant` (first_order, second_order, integrator_delay), `kp`, `ki`, `kd` | `PIDTuningDemo` |
| `time_response` | signals | `poles[]`, `zeros[]`, `gain`, `kind` (step, impulse) | `StepResponsePlot` |
| `bode_plot` **R** | signals | `poles[]`, `zeros[]`, `gain`, `asymptotes` | `BodePlot` |
| `nyquist_plot` | signals | `poles[]`, `zeros[]`, `gain` | `NyquistPlot` |
| `root_locus` | signals | `poles[]`, `zeros[]`, `k_max` | `RootLocus` |
| `pole_zero_map` | signals | `domain` (s, z), `view` (plane, surface, response), `poles[]`, `zeros[]` | `PoleZeroPlot`, `LaplaceSurface`, `PoleZeroToResponse` |
| `fourier_transform` | signals | `view` (winding, spectrum, series), `waveform`, `terms`, `frequency` | `WindingFourier`, `SpectrumMorph`, `FourierSeriesBuilder` |
| `waveform_generator` | signals | `kind` (sine, square, sawtooth, triangle), `frequency`, `amplitude`, `phase`, `show_phasor` | `WaveGenerator`, `PhasorWheel`, `PhasorSum` |
| `antenna_pattern` | signals | `antenna` (short_dipole, half_wave, isotropic), `dims` (2d, 3d), `show_fields` | `DipoleAntenna`, `RadiationPattern2D`, `RadiationPattern3D`, `EMWave`, `FieldRadiation` |
| `smith_chart` | signals | `z0`, `load_r`, `load_x`, `line_length_wl`, `show_swr` | `SmithChart`, `ImpedanceLocus` |
| `transmission_line` | signals | `view` (pulse, standing_wave), `z0`, `load_r`, `load_x` | `TransmissionLine`, `StandingWaveEnvelope` |
| `digital_filter` | signals | `kind` (fir, iir), `coefficients[]`, `view` (structure, response, both) | `TapDelayLine`, `IIRBiquad`, `PoleZeroToResponse` |
| `fft_butterfly` | signals | `n` (4, 8, 16), `stage` | `FFTButterfly` |
| `sampling_aliasing` | signals | `signal_hz`, `sample_hz` | `SamplingDemo` |
| `power_grid` | electrical | `layout` (radial, ring, substation), `show` (flows, fault), `fault_at`, `voltage_kv` | `PowerGrid`, `Substation` |
| `dc_dc_converter` | electrical | `topology` (buck, boost, buck_boost), `vin`, `duty`, `load_ohms`, `view` (schematic, waveforms, both) | `ConverterTopology`, `InductorRamp`, `ConversionRatioCurve` |
| `h_bridge` | electrical | `modulation` (bipolar, unipolar), `modulation_index`, `load` | `HBridge` |
| `three_phase_inverter` | electrical | `dc_volts`, `modulation_index`, `carrier_ratio`, `view` (pwm, filtered) | `ThreePhaseInverter`, `PWMToSine` |
| `pwm_comparator` | electrical | `reference` (sine, dc), `amplitude`, `carrier_ratio` | `PWMGenerator` |
| `foc_transform` | electrical | `view` (clarke, park, dq_frame, block_diagram), `frequency_hz` | `ClarkeParkTransform`, `DQFrame`, `TorqueDecomposition`, `FOCBlockDiagram` |
| `damper_windings` | electrical | `poles`, `slip` | `DamperWindings` |
| `swing_rotor_angle` | electrical | `inertia_h`, `mechanical_power`, `damping`, `view` (power_angle, time) | `RotorAngleOscillation` |
| `equal_area_criterion` | electrical | `mechanical_power`, `pmax_pre`, `pmax_fault`, `pmax_post`, `clearing_angle_deg` | `EqualAreaCriterion` |
| `fault_current` | electrical | `x_subtransient`, `x_transient`, `x_synchronous`, `dc_offset` | `FaultCurrentTransient` |
| `option_payoff` | finance | `kind` (call, put), `strategy` (single, straddle, bull_spread, collar), `strike`, `premium`, `side` | `OptionPayoff`, `OptionStrategy` |
| `option_greeks` | finance | `greek` (delta, gamma, theta, vega), `option_type`, `strike`, `volatility`, `view` (surface, curves) | `GreeksSurface`, `GreeksPanel`, `OptionValueCurve` |
| `exotic_option_paths` | finance | `kind` (up_and_out, down_and_out, asian), `barrier`, `strike`, `n_paths` | `BarrierOptionPaths`, `AsianOptionPath` |
| `garch_volatility` | finance | `view` (bands, recursion, shock), `alpha`, `beta`, `shock_at` | `GARCHSeries`, `GARCHRecursion`, `VolatilityShock` |
| `vol_surface` | finance | `level`, `skew`, `term_slope` | `VolSurfaceWireframe` |
| `order_book_matching` | finance | `scenario` (market_buy, partial_fill, limit_joins_queue), `depth` | `OrderBookStack`, `MatchingEngine`, `TradeTape`, `DepthChart` |
| `bond_price_yield` | finance | `view` (price_yield, duration_convexity), `coupon`, `maturity`, `yield_pct` | `BondPriceYieldCurve`, `CashflowLadder` |
| `yield_curve_shift` | finance | `shift` (parallel_up, steepen, flatten, invert), `size_bps` | `YieldCurve` |
| `brownian_motion` | finance | `variant` (random_walk, brownian, gbm), `paths`, `volatility`, `drift` | `BrownianPaths`, `RandomWalkToBrownian`, `GBMFan` |
| `binomial_tree` | finance | `steps`, `up`, `down`, `rate`, `strike` | `BinomialTree`, `BinomialConvergence` |
| `bloch_sphere` | quantum | `gates[]` (validated gate names), `start` (zero, one, plus, minus), `show_trail` | `BlochSphere`, `StateVector`, `MixedBlochBall` |
| `quantum_circuit` | quantum | `preset` (bell, ghz, superposition, custom), `gates[]`, `shots` | `QuantumCircuit`, `AmplitudeBars`, `ProbabilityHistogram`, `EntanglementDemo` |
| `phase_portrait` | math | `system` (preset or `custom`), `dx`, `dy`, `show_nullclines` | `PhasePortrait`, `VectorField2D`, `StreamLines`, `LimitCycleDemo` |
| `linear_transformation` **R** | math | `matrix[4]`, `show_eigenvectors`, `show_determinant` | `TransformableGrid`, `EigenvectorOverlay` |
| `eigen_decomposition` | math | `matrix[4]`, `view` (eigen, decomposition, svd) | `EigenDecompositionStory`, `SVDStory` |

The Rotating Magnetic Field has no new adapter: `RotatingMagneticField` is a Mobject for use inside other scenes, and
the existing `rotating_field` component already covers the standalone scene.

### 12.2 Adding an adapter (reference shape)

```python
@component(
    "gradient_descent",
    "A ball rolling downhill on a loss surface, with its gradient and momentum arrows; compares optimizers.",
    {"landscape": P("string", "terrain", default="ravine", choices=["bowl", "ravine", "saddle", "two_minima", "beale"]),
     "optimizer": P("string", "which optimizer to show", default="race", choices=["sgd", "momentum", "adam", "race"]),
     "learning_rate": P("number", "step size, 0.01 to 1", default=0.1, validator=_in_range(0.01, 1.0)),
     "start": P("number[]", "starting point [x, y]", default=[-1.6, 2.2], validator=_two_numbers)},
    {"landscape": "ravine", "optimizer": "race", "learning_rate": 0.05},   # example = test
    domain="machine-learning",
)
def gradient_descent(ctx: Ctx, landscape="ravine", optimizer="race", learning_rate=0.1, start=None) -> None:
    play_gradient_descent(ctx, landscape=landscape, optimizer=optimizer, lr=learning_rate, start=start)
```

### 12.3 Catalog budget

Today the planner prompt contains the whole catalog: 44 components, about 36 KB, roughly 830 bytes per entry. Adding the
adapters above would add on the order of **45 KB** (about 2.2 times the current catalog). A small planner model reads that
on every request and pays for it in latency, cost and accuracy, so the library must not simply be switched on.

| Option | Idea | Trade-off |
|--------|------|-----------|
| A. Ungated | Ship everything in the one catalog. | Simplest; prompt more than doubles; planner confusion grows. |
| **B. Domain-gated (recommended)** | Each component already carries a `domain` tag. The planner's existing pre-pass (the teaching brief in `docs/ARCHITECTURE.md`, or a cheap classification call if that pass is off) also names the 1 to 2 domains the topic needs; the storyboard prompt then lists `general` + `learn` + those domains only (about 20 to 30 entries). | One small worker change in the place `usableFor` already filters; deterministic; easy to test with golden prompts. Needs a fallback when the domain guess is wrong (include `general` and the closest domain, and let the planner ask). |
| C. Two-pass index | Prompt holds name plus one line for everything; a second call fetches full parameter specs for the shortlist. | Scales to hundreds of components; adds a model call per video. |

**Proposal:** Option B, delivered as its own behaviour-neutral PR **before** the first adapter is merged into the catalog,
so the prompt never grows. The renderer and the validation path are unchanged either way: all adapters are always valid
in the renderer, and the worker only decides which ones the model is shown.

### 12.4 Example storyboard scene

```json
{"component": "gradient_descent",
 "params": {"landscape": "ravine", "optimizer": "race", "learning_rate": 0.05},
 "narration": "Think of the loss as terrain. Plain gradient descent zigzags across the narrow valley, momentum smooths the zigzag, and Adam adapts its step to each direction."}
```

---

## 13. The 3Blue1Brown style guide, made testable

Each principle from the brief is turned into a rule with an automated check wherever one is possible. Review catches the rest.

| Principle | What it means in code | Check |
|-----------|----------------------|-------|
| **Visual intuition first** | Every Mobject docstring begins with an `Intuition:` line naming the geometric or physical picture. Every adapter description names the picture and the idea. Every domain PR answers the five design questions (Teaches, Params, Layout, Beats, Failure cases). | Test: every public Mobject has `Intuition:` in its docstring. PR template item. |
| **Smooth morphing and updaters** | State changes use `Transform`, `ReplacementTransform`, `ValueTracker`-driven updaters (`bind`), `LiveCurve`, `CurveMorph`. Cuts and "pop-in" are for the opening frame only. | `RecordingCtx` dry-run: at least one continuous animation (transform, updater or tracker) per scene, and `fade_in_whole` takes at most 2 s of the run. |
| **High-contrast dark mode palette** | Vectors blue / yellow / teal, heatmaps from `colormap`, glows only to direct attention. Black background. | Lint: no hex literals or `manim` colour constants outside `viz/core/palette.py` and `theme.py`. Contrast test: every text colour is at least 4.5:1 against `BG`. At most 3 simultaneous `Glow`/`FocusRing` objects. |
| **Colour is never the only signal** | Heatmaps offer `show_values`; hit/miss, stable/unstable and positive/negative carry a label or shape too. | Review item; `show_values` default on for grids at most 6 x 6. |
| **Modular, extensible API** | Keyword-only options with defaults; arrays and callables accepted; no Mobject touches `Scene` or `Ctx`; adapters have at most 5 params. | Test with `inspect`: signature shape, param counts, and that importing a Mobject module does not import `Scene` machinery. |
| **Stay on the stage** | Everything placed with `ctx.fit`. | Dry-run test: bounding box of the finished frame lies inside the stage rectangle. |
| **Fill the narration time** | Last beat uses `ctx.remaining`. | Dry-run test: `abs(ctx.elapsed - duration) <= 0.25`. |
| **Plain-word labels** | Text goes through `label()` / `MathLabel`; under 28 characters where possible. | Test: collected `Text` strings are under 40 characters, or wrapped. |
| **Readable at the product's quality** | Minimum stroke width 2 px and font size 20 at the `low` (480p15) preset. | Slow test renders one frame per domain; automatic check that the frame is not blank, plus a manual look at PR time. |

---

## 14. Testing, performance and delivery plan

### 14.1 Test tiers

| Tier | What | Where | Runs |
|------|------|-------|------|
| T0 | **Pure maths**, no Manim | `tests/viz/<domain>/test_math.py` | every PR, fast |
| T1 | **Build**: construct each Mobject, no render; check `.data`, counts, bounding boxes, caps | `tests/viz/<domain>/test_mobjects.py` | every PR, fast |
| T2 | **Dry run**: each `play_*` against a real `Scene` with `RecordingCtx` (elapsed time, animation classes, stage fit) | `tests/viz/<domain>/test_scenes.py` | every PR, fast |
| T3 | **Slow render**: at least one real clip per domain at 480p15, with a non-blank frame check | `-m slow` | CI `slow` job |
| T4 | **Catalog sync**: `python -m manimate.catalog` diffed against `worker/src/catalog.json` | existing CI step | every PR |

Examples of T0 invariants (each domain PR must include its own):

- **ML:** softmax rows sum to 1; causal mask zeroes the future; `alpha_bar` decreases monotonically; Adam/momentum paths match a hand-computed first two steps; backprop gradients match finite differences.
- **Microelectronics:** depletion width increases monotonically as bias goes from forward to reverse; MOSFET region boundaries at `vds = vgs - vth`; the cache simulator reproduces a hand-traced access sequence; CAN and USB bit stuffing never leaves six equal bits in a row; SPI sampling instants per mode.
- **Signals:** Bode asymptotes agree with exact magnitude far from corners; step-response overshoot matches the second-order closed form; the winding centroid peaks at the signal's true frequency; reflection coefficient and SWR identities; FFT butterfly output equals `numpy.fft.fft`.
- **Power:** steady-state `Vout = D * Vin` for the buck and `Vin / (1 - D)` for the boost; Clarke then Park of a balanced three-phase set gives constants; equal-area `A_acc = A_dec` at the critical clearing angle; the swing simulation conserves energy when `D = 0`.
- **Finance:** put-call parity; call delta in (0, 1); gamma positive; binomial price converges to Black-Scholes; order matching conserves quantity and respects price-time priority; bond price falls as yield rises and convexity is positive.
- **Quantum and math:** each gate matrix is unitary; `H` rotates about `(x+z)/sqrt(2)` by `pi`; reduced Bloch vector has length 1 for a product state and 0 for a Bell state; `A v = lambda v` for every returned eigenpair; `U S Vt` reconstructs the matrix; fixed-point classification matches the Jacobian's eigenvalues.

### 14.2 Performance budgets (targets, calibrated in PR 1)

These are starting targets to be measured with `renderer/scripts/profile_render.py` and then fixed as constants in `viz/core`.

| Budget | Target |
|--------|--------|
| Visible VMobjects per frame | about 1,500 at most |
| `Surface2p5D` resolution | at most 32 x 32 quads; animated morphs precompute frames with vectorised NumPy |
| `HeatmapGrid` | cells above 16 x 16 become one `ImageMobject` |
| `ParticleField` | at most 80 particles |
| Active updaters | about 40 at most |
| Wall-clock for a 20 s clip at 480p15 on the CI runner | about 60 s at most (to be confirmed) |
| `MathLabel`s per scene | about 6 at most (each is a LaTeX compile) |

### 14.3 Delivery plan

Every PR is its own review, regenerates the catalog, and carries tests and example scripts. A PR that exceeds about 800
changed non-test lines is split.

| PR | Title | Contents |
|----|-------|----------|
| **0** | `docs: viz architecture README` | **This file only.** |
| 1 | `viz: core` | Section 5 in full, `VizScene`, `RecordingCtx`, style lints, performance constants, `safe_expr` multi-variable extension. No catalog change. |
| 2 | `worker: domain-gated catalog` | Option B of section 12.3 with golden-prompt tests. Behaviour-neutral for the current catalog. |
| 3a-3c | `viz: ml_advanced` | 3a attention, positional encoding, KV cache; 3b diffusion and VAEs; 3c optimization and backprop. Adapters, examples, catalog. |
| 4a-4c | `viz: microelectronics` | 4a PN junction and MOSFET; 4b pipeline, branch prediction, caches, DMA; 4c bus protocols. |
| 5a-5d | `viz: signals` | 5a block diagrams, PID, responses, Bode, Nyquist, root locus; 5b waveforms, Fourier, Laplace surface; 5c RF and transmission lines; 5d DSP. |
| 6a-6c | `viz: power` | 6a converters, H-bridge, PWM, inverter; 6b machines, `d-q`, FOC; 6c grid one-lines and transients. |
| 7a-7b | `viz: finance` | 7a Greeks, exotics, GARCH, vol surface; 7b order book, bonds, yield curves, stochastic processes. |
| 8a-8b | `viz: quantum_math` | 8a Bloch sphere and circuits; 8b dynamical systems and linear algebra. |
| 9 | `docs: contributor page` | Web contributor page, `docs/ARCHITECTURE.md` packs list, roadmap rows marked as owned by `viz`. |
| 10 (optional) | `renderer: native 3D path` | Only if section 15 question 2 concludes that 2.5D is not enough. |

**Definition of done (every PR)**

- [ ] The five design questions are answered in the PR description for every new component
- [ ] Pure-math functions with T0 tests, including the domain invariants above
- [ ] Mobjects with `Intuition:` docstrings, `.data`, `bind`, and T1 tests
- [ ] `play_*` scenes with T2 dry-run tests and at least one T3 render per domain
- [ ] `@component` adapters with ranges, defaults, a rendering `example`, and plain-English errors
- [ ] An example script in `renderer/examples/viz/` and (for adapters) a JSON storyboard
- [ ] One rendered clip watched by eye: text on stage, nothing clipped, readable at 480p
- [ ] `worker/src/catalog.json` and `web/lib/catalog.json` regenerated
- [ ] A domain-knowledge review of the physics, circuits or finance by someone who knows the field

---

## 15. Risks and open questions

Questions are ordered by how much code they change. Each has a recommendation, so a reviewer can simply agree or overrule.

1. **Where should the library live?** Options: a subpackage `manimate.viz` (proposed), or a separate installable distribution for the pure Mobject layer. *Recommendation:* subpackage now; the layers are already separated, so extracting `viz.core` and the domains later is mechanical.
2. **3D strategy.** The renderer's scene is a plain 2D `Scene`. *Recommendation:* use the 2.5D projection (`Projector3D`) everywhere in this series. A native `ThreeDScene` path means changing the scene class, the interpreter and the caption layout, and is only justified if the surfaces (loss landscapes, Greeks, radiation lobes) prove unconvincing in review. Reviewers: please judge the 2.5D look from the PR 1 sample renders.
3. **Catalog growth.** About 45 KB more catalog (section 12.3). *Recommendation:* Option B, landing before any adapter reaches the catalog.
4. **Scope.** This is a large surface: about 170 Mobjects, 55 scene functions and 55 adapters. *Mitigation:* the ordering in 14.3 lets the project stop after any domain with something complete. Items most easily cut or deferred if time is short: `Spectrogram`, `FieldRadiation`, `PowerIterationDemo`, `CPUDatapath`, `HazardTimeline`, `GBMFan`, `BinomialConvergence`. *Backlog* (explicitly out of this series): CMOS inverter and BJT, space-vector PWM hexagon, Nichols chart, stub matching, polarization ellipse, LSTM/GRU gates, state-space models, Heston volatility, Monte Carlo variance reduction.
5. **LaTeX.** Maths labels compile LaTeX per label, which costs render time, and LaTeX may be absent in some environments. *Recommendation:* `MathLabel` everywhere (graceful text fallback), a cap of about six per scene, and a check in PR 1 that the production image has LaTeX.
6. **Correctness.** Physics, circuits and finance can be subtly wrong while looking convincing. *Recommendation:* every model is the simplest textbook one, states its assumptions on screen, is tested against a closed form (section 14.1), and gets a domain review before merge. Nothing is presented as advice or as a precise simulation.
7. **Legibility at the product's default quality (480p15).** Thin wireframes, small tick labels and glows can disappear at low resolution. *Recommendation:* minimum stroke and font sizes in `viz.core`, verified on a rendered frame in each domain's T3 test.
8. **Overlap with shipped components.** Section 2.3 keeps all 44 shipped components and adopts four reserved roadmap names. *Question:* confirm the series may own roadmap items 17, 32, 46 and 50.
9. **Naming.** `ml_advanced` follows the brief's `manim_ml_advanced`; the other five packages are named after their field. *Question:* keep these names, or prefer one convention (for example `viz.dl`, `viz.micro`, `viz.dsp`, `viz.power`, `viz.fin`, `viz.qm`)?
10. **Originality and licensing.** The 3Blue1Brown *aesthetic* is a style, not code. Everything is implemented from scratch for this repository, and no snippets are to be copied from other video projects, whose licences may not be compatible with this repo's MIT licence.

---

## 16. Appendix: traceability from the brief

Every bullet in the brief maps to at least one Mobject and one scene or adapter.

| Brief item | Mobjects and utilities | Scene / adapter |
|------------|-----------------------|-----------------|
| **1. Attention heads: QKV grids, live dot-product heatmaps** | `QKVProjection`, `AttentionHeatmap` (`animate_dot_product`), `AttentionHead`, `MultiHeadAttention`, `AttentionArcs` | `attention_head`, `attention_heatmap` |
| Positional encoding: wave summation on embeddings | `PositionalEncodingWaves`, `PositionalEncodingSum`, `PEHeatmap` | `positional_encoding` |
| KV cache: sliding-window buffers | `KVCacheBuffer` (`window`, `animate_slide`), `DecodeTimeline` | `kv_cache` |
| Forward/reverse noising over 2D latent grids | `LatentGrid`, `SchedulePlot`, `ForwardDiffusion`, `ReverseDiffusion`, `Distribution2DDiffusion`, `ScoreField` | `diffusion_process` |
| Autoencoders / VAEs, latent interpolation curves | `Autoencoder`, `LatentSpace2D`, `LatentInterpolation`, `VAESampling` | `autoencoder_latent` |
| Loss landscapes with SGD / Adam / Momentum paths and momentum vectors | `LossLandscape`, `DescentPath`, `OptimizerRace`, `LearningRateStrip`, `LossCurve` | `gradient_descent` |
| Backprop flow: reverse glowing pulses carrying `dL/dw` | `BackpropNetwork` (`animate_backward`), `ChainRuleStrip`, `ComputationGraph`, `PulseTrain` | `backprop_flow` |
| **2. PN junctions: recombination, depletion width under bias** | `PNJunction`, `PNBandDiagram`, `DiodeIVCurve`, `ParticleField` | `pn_junction` |
| MOSFET channel formation, `I_D` velocity vectors | `MOSFETCrossSection`, `DrainCurrentVectors`, `MOSFETCharacteristics`, `TransferCurve` | `mosfet_channel` |
| CPU pipeline with hazard stalls and branch prediction | `PipelineDiagram`, `HazardTimeline`, `BranchPredictionPath`, `PredictorFSM`, `CPUDatapath` | `cpu_pipeline`, `branch_prediction` |
| Cache hierarchy hit/miss, address buses, DMA | `CacheArray`, `CacheHierarchy`, `AddressBus`, `DataBus`, `RAMArray`, `DMAController` | `cache_access`, `memory_system` |
| SPI / I2C / UART frames, `D+`/`D-`, CAN | `DigitalWaveform`, `ClockLine`, `SPIBus`, `I2CBus`, `UARTFrame`, `DifferentialPair`, `USBLine`, `CANBus`, `CANArbitration` | `serial_bus` |
| **3. Block diagrams: `H(s)`, summing, feedback, PID, state space** | `TransferFunctionBlock`, `SummingJunction`, `FeedbackLoop`, `PIDController`, `StateSpaceDiagram`, `BlockDiagram`, `Block`, `Wire` | `block_diagram`, `pid_response` |
| Waveform generators, phasors, Bode, Nyquist, impulse/step | `WaveGenerator`, `PhasorWheel`, `PhasorSum`, `BodePlot`, `NyquistPlot`, `RootLocus`, `StepResponsePlot` (`kind`: step, impulse) | `waveform_generator`, `bode_plot`, `nyquist_plot`, `root_locus`, `time_response` |
| Time-to-frequency morphs: Fourier and Laplace | `WindingFourier`, `SpectrumMorph`, `FourierSeriesBuilder`, `LaplaceSurface`, `Spectrogram` | `fourier_transform`, `pole_zero_map` |
| Antenna patterns 2D/3D, animated `E` and `H` fields | `DipoleAntenna`, `RadiationPattern2D`, `RadiationPattern3D`, `EMWave`, `FieldRadiation` | `antenna_pattern` |
| Smith chart, `Gamma`, SWR envelopes | `SmithChart`, `ImpedanceLocus`, `StandingWaveEnvelope`, `TransmissionLine` | `smith_chart`, `transmission_line` |
| FIR/IIR taps, `Z`-plane to frequency response, FFT butterfly | `TapDelayLine`, `IIRBiquad`, `PoleZeroPlot`, `PoleZeroToResponse`, `FFTButterfly`, `SamplingDemo` | `digital_filter`, `pole_zero_map`, `fft_butterfly`, `sampling_aliasing` |
| **4. Grid components: lines, transformers, generators, buses** | `Bus`, `Breaker`, `TransmissionLineSymbol`, `TransformerSymbol`, `SynchronousGeneratorSymbol`, `Substation`, `PowerGrid` | `power_grid` |
| Buck / boost / buck-boost / H-bridge, PWM, `di/dt` ramps | `ConverterTopology`, `InductorRamp`, `ConversionRatioCurve`, `HBridge`, `PWMGenerator` | `dc_dc_converter`, `h_bridge`, `pwm_comparator` |
| 3-phase PWM: DC bus to clean 3-phase AC | `ThreePhaseInverter`, `PWMToSine` | `three_phase_inverter` |
| Field-Oriented Control, `d-q` frames | `DQFrame`, `ClarkeParkTransform`, `TorqueDecomposition`, `FOCBlockDiagram`, `Stator3Phase`, `RotorFrame` | `foc_transform` |
| Damper windings, RMF, rotor angle, fault transients | `DamperWindings`, `RotatingMagneticField`, `RotorAngleOscillation`, `EqualAreaCriterion`, `FaultCurrentTransient` | `damper_windings`, `swing_rotor_angle`, `equal_area_criterion`, `fault_current` (RMF via existing `rotating_field`) |
| **5. Option Greeks surfaces** | `GreeksSurface`, `GreeksPanel`, `OptionValueCurve` | `option_greeks` |
| GARCH and volatility clustering | `GARCHSeries`, `GARCHRecursion`, `VolatilityShock`, `AreaBand` | `garch_volatility` |
| Exotic options | `BarrierOptionPaths`, `AsianOptionPath`, `OptionPayoff`, `OptionStrategy` | `exotic_option_paths`, `option_payoff` |
| Order book matching engine | `OrderBookStack`, `MatchingEngine`, `TradeTape`, `DepthChart` | `order_book_matching` |
| Yield curves, price-yield, convexity | `BondPriceYieldCurve`, `CashflowLadder`, `YieldCurve` | `bond_price_yield`, `yield_curve_shift` |
| Brownian motion, binomial trees, vol surface | `BrownianPaths`, `RandomWalkToBrownian`, `GBMFan`, `BinomialTree`, `BinomialConvergence`, `VolSurfaceWireframe` | `brownian_motion`, `binomial_tree`, `vol_surface` |
| **6. Bloch sphere with `X, Y, Z, H` rotations** | `BlochSphere`, `StateVector` (`animate_gate`, `animate_measure`), `MixedBlochBall`, `Sphere2p5D` | `bloch_sphere` |
| Quantum circuits, CNOT entanglement, probability histograms | `QuantumCircuit`, `AmplitudeBars`, `ProbabilityHistogram`, `EntanglementDemo` | `quantum_circuit` |
| Vector fields and phase portraits | `VectorField2D`, `StreamLines`, `PhasePortrait`, `JacobianLens`, `LimitCycleDemo` | `phase_portrait` |
| Eigen-decompositions | `TransformableGrid`, `EigenvectorOverlay`, `EigenDecompositionStory`, `SVDStory`, `PowerIterationDemo` | `linear_transformation`, `eigen_decomposition` |
| **Phase 3: style guide** | Section 13, `viz.core` (`PALETTE`, `Glow`, `FocusRing`, `fade_in_whole`, `bind`, `LiveCurve`) | enforced by the T0 to T2 checks |
