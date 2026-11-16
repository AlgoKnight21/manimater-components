# Third-party Manim libraries

Some components draw with community Manim plugins instead of hand-written Manim (see `renderer/README.md`).
Only libraries under the MIT licence (or a Creative Commons licence) are used. They are installed as dependencies by
`renderer/scripts/install_plugins.sh`, not copied into this repository.

## Used

| Library | Version | Licence | Source of the licence | Components |
|---|---|---|---|---|
| [manim-physics](https://github.com/Matheart/manim-physics) | 0.4.0 | MIT | LICENSE file | `electric_field`, `standing_wave`, `pendulum`, `falling_objects`, `lens_rays` |
| [ManimML](https://github.com/helblazer811/ManimML) | 0.0.24 | MIT | LICENSE file | `neural_network` |
| [manim-dsa](https://github.com/F4bbi/manim-dsa) | 0.4.0 | MIT | `pyproject.toml`, README and PyPI declare MIT; the repository has no LICENSE file | `array_sort`, `stack_operations`, `graph_traversal`, `binary_search_tree` |
| [manim-chemistry](https://github.com/UnMolDeQuimica/manim-chemistry) | 0.5.1 | MIT | LICENSE file | `bohr_atom`, `molecule_structure`, `chemical_equation` |
| [manim-databases](https://github.com/pg-pilot/manim-databases) | commit `dee0ae1` | MIT | LICENSE file | `database_table`, `btree_index` |
| [manim-eng](https://github.com/overegneered/manim-eng) | 0.1.0 | MIT, with an extra clause (below) | LICENSE file | `circuit_diagram` |
| [manim-extensions](https://github.com/ExploreMaths/manim_extensions) | 1.0.5 | MIT | LICENSE file and PyPI metadata | `supply_demand`, `ad_as_model`, `is_lm_model`, `is_lm_ad_as`, `solow_growth`, `concept_map`, `hierarchy_tree`, `timeline`, `file_tree`, `gear_train`, `queue_operations` |

manim-databases is early-alpha and not on PyPI, so it is installed from the pinned commit above.

The website's Components page credits these libraries (name, link, licence and the components that use each). That list
comes from `renderer/manimate/plugins.py`, and a test checks it against the table above.

### manim-eng's extra clause

manim-eng is MIT-licensed, but its author adds that he revokes any right to use the repository's contents for training AI
models without his prior written approval. The maintainers decided that rendering circuit diagrams with the library is
not model training, so it is used. The condition that follows: nothing from manim-eng (its source, documentation or
examples) may be used as training or fine-tuning data, and it should not be copied into this repository. The planner
prompt only contains this project's own component descriptions.

### How manim-extensions is used

manim-extensions bundles many small Manim libraries, so it is used through one generic adapter, `renderer/manimate/components/_ext.py`:
each component is a declaration of what to build and which animation steps to play, and one runner handles layout, timing and
the fallback. Only the economics, mind-map, file-tree, gear and queue parts are exposed so far. Parts left out on purpose:
the periodic table (about 40 s to build, too slow for a video), the Rubik's cube (needs a 3D scene), and the decision tree
(needs a leaf image per class and scikit-learn).

## Evaluated and not used

| Library | Licence | Why not |
|---|---|---|
| manim-chessrender | No licence found | Cannot be used without a licence |
| statanim | MIT | Built for 3D scenes: its README example fails on the current release, and in a 2D scene its curve collapses to a line and its Venn diagram is empty |
| manim-automata | MIT | `play_string` crashes (`UnboundLocalError`) and its JSON input path is broken |
| manim-algorithm | MIT | Fails to import (`NameError`) |
| manim-circuit | MIT | Works, but the existing `circuit` component already draws closed loops with animated current, and manim-eng has more symbols |
| manim-sequence-diagram | MIT | Works, but small and rough: labels overlap lines and the diagram cannot be rescaled |
| manim-Astronomy | MIT | Not tested: requires `vedo`, a heavy 3D toolkit |
| manimextra | MIT | Not tested: its metadata requires Python 3.12 (the renderer runs 3.11) |
| manim-pymunk | MIT (GitHub; none on PyPI) | Not tested: overlaps manim-physics, which is already used |
| manim-data-structures, chanim | MIT | Not tested: overlap manim-dsa and manim-chemistry |
| manim-voiceover, manim-slides | MIT | Not component libraries (speech and slide export) |

The six libraries in use were checked against Manim 0.22 on Python 3.11 and 3.12. Entries marked "Not tested" were
rejected on their metadata or purpose alone.
