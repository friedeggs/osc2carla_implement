# Third-party material

## PMSF py-osc2 (OpenSCENARIO 2.1 grammar)

Vendored under `third_party/py-osc2/` and copied into `grammar/openscenario2.g4`.

The ANTLR lexer/parser in `osc2carla/frontend/generated/` is generated from that grammar. This package does **not** import `osc2parser` at runtime.

- Upstream: https://github.com/PMSFIT/py-osc2
- License: Mozilla Public License 2.0 (`third_party/py-osc2/LICENSE`)

## CARLA 0.9.16

Runtime dependency for live simulation (UE4 server + Python API). Not vendored here.

- https://github.com/carla-simulator/carla
- Use the 0.9.16 Linux server and the `cp38` Python wheel.

## Python libraries

Installed via `requirements.txt`: `antlr4-python3-runtime` (4.7.x), `py-trees` 0.8.3, `numpy`, and optionally `opencv-python` for video overlays. `ffmpeg` is invoked as an external binary to encode MP4s.

CARLA Scenario Runner is **not** a dependency of this compiler and is not included.
