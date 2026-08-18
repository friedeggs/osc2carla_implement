# Town `loop`

Single rectangular circuit with long straights; car-following demos never run out of road. No four-way junction -- every corner is a two-arm turn.

- extent: x in [-5.2, 245.2], y in [-5.2, 145.2]
- lanes: 32 (including junction connectors)
- spawn points: 144
- lane width: 3.5 m, 2 lanes each way

Coordinates are CARLA's: +x east, +y **south**, `h` in radians growing clockwise when the map is drawn with +y down. A left turn goes toward -y.

## Junctions

`arms` lists the compass directions that have a road. Only a junction with all four (`NESW`) can stage a conflict that needs an opposing approach, such as an unprotected left.

| centre (x, y) | arms | marker to use as a distance reference |
|---|---|---|
| (0, 0) | `ES` | `position(x: 0.00, y: 0.00, z: -0.40, at: start)` |
| (0, 140) | `NE` | `position(x: 0.00, y: 140.00, z: -0.40, at: start)` |
| (240, 0) | `SW` | `position(x: 240.00, y: 0.00, z: -0.40, at: start)` |
| (240, 140) | `NW` | `position(x: 240.00, y: 140.00, z: -0.40, at: start)` |

## Carriageways

A vehicle placed on one of these, with the matching `h`, drives along it.

| runs along | fixed coord | direction | lane | `h` (rad) | travel range |
|---|---|---|---|---|---|
| x | y = -5.25 | west (-x) | +2 | +3.1416 | x in [11, 229] |
| x | y = -1.75 | west (-x) | +1 | +3.1416 | x in [11, 229] |
| x | y = 1.75 | east (+x) | -1 | +0.0000 | x in [11, 229] |
| x | y = 5.25 | east (+x) | -2 | +0.0000 | x in [11, 229] |
| x | y = 134.75 | west (-x) | +2 | +3.1416 | x in [11, 229] |
| x | y = 138.25 | west (-x) | +1 | +3.1416 | x in [11, 229] |
| x | y = 141.75 | east (+x) | -1 | +0.0000 | x in [11, 229] |
| x | y = 145.25 | east (+x) | -2 | +0.0000 | x in [11, 229] |
| y | x = -5.25 | south (+y) | -2 | +1.5708 | y in [11, 129] |
| y | x = -1.75 | south (+y) | -1 | +1.5708 | y in [11, 129] |
| y | x = 1.75 | north (-y) | +1 | -1.5708 | y in [11, 129] |
| y | x = 5.25 | north (-y) | +2 | -1.5708 | y in [11, 129] |
| y | x = 234.75 | south (+y) | -2 | +1.5708 | y in [11, 129] |
| y | x = 238.25 | south (+y) | -1 | +1.5708 | y in [11, 129] |
| y | x = 241.75 | north (-y) | +1 | -1.5708 | y in [11, 129] |
| y | x = 245.25 | north (-y) | +2 | -1.5708 | y in [11, 129] |

`lane -1` is the lane nearest the road's centre line and `lane -2` the one outside it, following CARLA's sign convention; the positive ids are the opposing carriageway.

Travel ranges stop at the junction boxes: the segment between two junctions is where a vehicle has straight road. Placing an actor outside every range puts it off the network, where `get_waypoint(project_to_road=True)` will snap it to whatever lane happens to be nearest.
