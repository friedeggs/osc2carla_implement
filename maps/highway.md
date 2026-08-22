# Town `highway`

Three lanes each way with ~470 m of straight between junctions -- the stand-in for the Town04 highway. The one to stage lane changes, cut-ins and any conflict that needs a lane on both sides of the ego.

- extent: x in [-8.8, 508.8], y in [-8.8, 168.8]
- lanes: 48 (including junction connectors)
- spawn points: 396
- lane width: 3.5 m, 3 lanes each way

Coordinates are CARLA's: +x east, +y **south**, `h` in radians growing clockwise when the map is drawn with +y down. A left turn goes toward -y.

## Junctions

`arms` lists the compass directions that have a road. Only a junction with all four (`NESW`) can stage a conflict that needs an opposing approach, such as an unprotected left.

| centre (x, y) | arms | marker to use as a distance reference |
|---|---|---|
| (0, 0) | `ES` | `position(x: 0.00, y: 0.00, z: -0.40, at: start)` |
| (0, 160) | `NE` | `position(x: 0.00, y: 160.00, z: -0.40, at: start)` |
| (500, 0) | `SW` | `position(x: 500.00, y: 0.00, z: -0.40, at: start)` |
| (500, 160) | `NW` | `position(x: 500.00, y: 160.00, z: -0.40, at: start)` |

## Carriageways

A vehicle placed on one of these, with the matching `h`, drives along it.

| runs along | fixed coord | direction | lane | `h` (rad) | travel range |
|---|---|---|---|---|---|
| x | y = -8.75 | west (-x) | +3 | +3.1416 | x in [14.5, 485.5] |
| x | y = -5.25 | west (-x) | +2 | +3.1416 | x in [14.5, 485.5] |
| x | y = -1.75 | west (-x) | +1 | +3.1416 | x in [14.5, 485.5] |
| x | y = 1.75 | east (+x) | -1 | +0.0000 | x in [14.5, 485.5] |
| x | y = 5.25 | east (+x) | -2 | +0.0000 | x in [14.5, 485.5] |
| x | y = 8.75 | east (+x) | -3 | +0.0000 | x in [14.5, 485.5] |
| x | y = 151.25 | west (-x) | +3 | +3.1416 | x in [14.5, 485.5] |
| x | y = 154.75 | west (-x) | +2 | +3.1416 | x in [14.5, 485.5] |
| x | y = 158.25 | west (-x) | +1 | +3.1416 | x in [14.5, 485.5] |
| x | y = 161.75 | east (+x) | -1 | +0.0000 | x in [14.5, 485.5] |
| x | y = 165.25 | east (+x) | -2 | +0.0000 | x in [14.5, 485.5] |
| x | y = 168.75 | east (+x) | -3 | +0.0000 | x in [14.5, 485.5] |
| y | x = -8.75 | south (+y) | -3 | +1.5708 | y in [14.5, 145.5] |
| y | x = -5.25 | south (+y) | -2 | +1.5708 | y in [14.5, 145.5] |
| y | x = -1.75 | south (+y) | -1 | +1.5708 | y in [14.5, 145.5] |
| y | x = 1.75 | north (-y) | +1 | -1.5708 | y in [14.5, 145.5] |
| y | x = 5.25 | north (-y) | +2 | -1.5708 | y in [14.5, 145.5] |
| y | x = 8.75 | north (-y) | +3 | -1.5708 | y in [14.5, 145.5] |
| y | x = 491.25 | south (+y) | -3 | +1.5708 | y in [14.5, 145.5] |
| y | x = 494.75 | south (+y) | -2 | +1.5708 | y in [14.5, 145.5] |
| y | x = 498.25 | south (+y) | -1 | +1.5708 | y in [14.5, 145.5] |
| y | x = 501.75 | north (-y) | +1 | -1.5708 | y in [14.5, 145.5] |
| y | x = 505.25 | north (-y) | +2 | -1.5708 | y in [14.5, 145.5] |
| y | x = 508.75 | north (-y) | +3 | -1.5708 | y in [14.5, 145.5] |

`lane -1` is the lane nearest the road's centre line and `lane -2` the one outside it, following CARLA's sign convention; the positive ids are the opposing carriageway.

Travel ranges stop at the junction boxes: the segment between two junctions is where a vehicle has straight road. Placing an actor outside every range puts it off the network, where `get_waypoint(project_to_road=True)` will snap it to whatever lane happens to be nearest.
