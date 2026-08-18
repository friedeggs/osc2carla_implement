# Town `grid`

3x3 junctions, 80 m spacing, two lanes each way. One four-way junction, at (80, 80).

- extent: x in [-5.3, 165.3], y in [-5.3, 165.3]
- lanes: 112 (including junction connectors)
- spawn points: 192
- lane width: 3.5 m, 2 lanes each way

Coordinates are CARLA's: +x east, +y **south**, `h` in radians growing clockwise when the map is drawn with +y down. A left turn goes toward -y.

## Junctions

`arms` lists the compass directions that have a road. Only a junction with all four (`NESW`) can stage a conflict that needs an opposing approach, such as an unprotected left.

| centre (x, y) | arms | marker to use as a distance reference |
|---|---|---|
| (0, 0) | `ES` | `position(x: 0.00, y: 0.00, z: -0.40, at: start)` |
| (0, 80) | `NES` | `position(x: 0.00, y: 80.00, z: -0.40, at: start)` |
| (0, 160) | `NE` | `position(x: 0.00, y: 160.00, z: -0.40, at: start)` |
| (80, 0) | `ESW` | `position(x: 80.00, y: 0.00, z: -0.40, at: start)` |
| (80, 80) | `NESW` | `position(x: 80.00, y: 80.00, z: -0.40, at: start)` |
| (80, 160) | `NEW` | `position(x: 80.00, y: 160.00, z: -0.40, at: start)` |
| (160, 0) | `SW` | `position(x: 160.00, y: 0.00, z: -0.40, at: start)` |
| (160, 80) | `NSW` | `position(x: 160.00, y: 80.00, z: -0.40, at: start)` |
| (160, 160) | `NW` | `position(x: 160.00, y: 160.00, z: -0.40, at: start)` |

## Carriageways

A vehicle placed on one of these, with the matching `h`, drives along it.

| runs along | fixed coord | direction | lane | `h` (rad) | travel range |
|---|---|---|---|---|---|
| x | y = -5.25 | west (-x) | +2 | +3.1416 | x in [7, 153] |
| x | y = -1.75 | west (-x) | +1 | +3.1416 | x in [7, 153] |
| x | y = 1.75 | east (+x) | -1 | +0.0000 | x in [7, 153] |
| x | y = 5.25 | east (+x) | -2 | +0.0000 | x in [7, 153] |
| x | y = 74.75 | west (-x) | +2 | +3.1416 | x in [7, 153] |
| x | y = 78.25 | west (-x) | +1 | +3.1416 | x in [7, 153] |
| x | y = 81.75 | east (+x) | -1 | +0.0000 | x in [7, 153] |
| x | y = 85.25 | east (+x) | -2 | +0.0000 | x in [7, 153] |
| x | y = 154.75 | west (-x) | +2 | +3.1416 | x in [7, 153] |
| x | y = 158.25 | west (-x) | +1 | +3.1416 | x in [7, 153] |
| x | y = 161.75 | east (+x) | -1 | +0.0000 | x in [7, 153] |
| x | y = 165.25 | east (+x) | -2 | +0.0000 | x in [7, 153] |
| y | x = -5.25 | south (+y) | -2 | +1.5708 | y in [7, 153] |
| y | x = -1.75 | south (+y) | -1 | +1.5708 | y in [7, 153] |
| y | x = 1.75 | north (-y) | +1 | -1.5708 | y in [7, 153] |
| y | x = 5.25 | north (-y) | +2 | -1.5708 | y in [7, 153] |
| y | x = 74.75 | south (+y) | -2 | +1.5708 | y in [7, 153] |
| y | x = 78.25 | south (+y) | -1 | +1.5708 | y in [7, 153] |
| y | x = 81.75 | north (-y) | +1 | -1.5708 | y in [7, 153] |
| y | x = 85.25 | north (-y) | +2 | -1.5708 | y in [7, 153] |
| y | x = 154.75 | south (+y) | -2 | +1.5708 | y in [7, 153] |
| y | x = 158.25 | south (+y) | -1 | +1.5708 | y in [7, 153] |
| y | x = 161.75 | north (-y) | +1 | -1.5708 | y in [7, 153] |
| y | x = 165.25 | north (-y) | +2 | -1.5708 | y in [7, 153] |

`lane -1` is the lane nearest the road's centre line and `lane -2` the one outside it, following CARLA's sign convention; the positive ids are the opposing carriageway.

Travel ranges stop at the junction boxes: the segment between two junctions is where a vehicle has straight road. Placing an actor outside every range puts it off the network, where `get_waypoint(project_to_road=True)` will snap it to whatever lane happens to be nearest.
