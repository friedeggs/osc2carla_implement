# Town `two_lane`

One lane each way, undivided, ~385 m of straight. The left neighbour of a lane here is oncoming traffic, which is what an overtake needs and a divided highway cannot provide.

- extent: x in [-1.8, 401.8], y in [-1.8, 121.8]
- lanes: 16 (including junction connectors)
- spawn points: 108
- lane width: 3.5 m, 1 lanes each way

Coordinates are CARLA's: +x east, +y **south**, `h` in radians growing clockwise when the map is drawn with +y down. A left turn goes toward -y.

## Junctions

`arms` lists the compass directions that have a road. Only a junction with all four (`NESW`) can stage a conflict that needs an opposing approach, such as an unprotected left.

| centre (x, y) | arms | marker to use as a distance reference |
|---|---|---|
| (0, 0) | `ES` | `position(x: 0.00, y: 0.00, z: -0.40, at: start)` |
| (0, 120) | `NE` | `position(x: 0.00, y: 120.00, z: -0.40, at: start)` |
| (400, 0) | `SW` | `position(x: 400.00, y: 0.00, z: -0.40, at: start)` |
| (400, 120) | `NW` | `position(x: 400.00, y: 120.00, z: -0.40, at: start)` |

## Carriageways

A vehicle placed on one of these, with the matching `h`, drives along it.

| runs along | fixed coord | direction | lane | `h` (rad) | travel range |
|---|---|---|---|---|---|
| x | y = -1.75 | west (-x) | +1 | +3.1416 | x in [7.5, 392.5] |
| x | y = 1.75 | east (+x) | -1 | +0.0000 | x in [7.5, 392.5] |
| x | y = 118.25 | west (-x) | +1 | +3.1416 | x in [7.5, 392.5] |
| x | y = 121.75 | east (+x) | -1 | +0.0000 | x in [7.5, 392.5] |
| y | x = -1.75 | south (+y) | -1 | +1.5708 | y in [7.5, 112.5] |
| y | x = 1.75 | north (-y) | +1 | -1.5708 | y in [7.5, 112.5] |
| y | x = 398.25 | south (+y) | -1 | +1.5708 | y in [7.5, 112.5] |
| y | x = 401.75 | north (-y) | +1 | -1.5708 | y in [7.5, 112.5] |

`lane -1` is the lane nearest the road's centre line and `lane -2` the one outside it, following CARLA's sign convention; the positive ids are the opposing carriageway.

Travel ranges stop at the junction boxes: the segment between two junctions is where a vehicle has straight road. Placing an actor outside every range puts it off the network, where `get_waypoint(project_to_road=True)` will snap it to whatever lane happens to be nearest.
