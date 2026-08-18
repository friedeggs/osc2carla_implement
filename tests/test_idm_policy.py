"""Simulator-free checks of the IDM policy and the boundary types."""
from osc2carla.backend.policy import (IDMPolicy, Leader, Observation, RoutePoint,
                                      resolve_policy, parse_policy_params)

p = IDMPolicy()
v0 = p.p["v0"]

# 1. free road, at rest -> accelerate at ~a_max
a = p.acceleration(0.0, None)
assert abs(a - p.p["a_max"]) < 1e-9, a
print(f"free road, v=0      -> a={a:+.3f} m/s^2  (expect +{p.p['a_max']})")

# 2. free road, at desired speed -> zero acceleration
a = p.acceleration(v0, None)
assert abs(a) < 1e-9, a
print(f"free road, v=v0     -> a={a:+.3f} m/s^2  (expect 0)")

# 3. free road, above desired speed -> decelerate
a = p.acceleration(v0 * 1.5, None)
assert a < 0, a
print(f"free road, v=1.5*v0 -> a={a:+.3f} m/s^2  (expect < 0)")

# 4. equilibrium: at the IDM equilibrium gap with matched speed, a == 0.
#    The equilibrium gap is s*(v,0)/sqrt(1-(v/v0)^delta), NOT s0+v*T -- those
#    coincide only when v << v0, so use the exact expression.
import math as _m
v = 6.0
free = 1.0 - (v / p.p["v0"]) ** p.p["delta"]
s_eq = (p.p["s0"] + v * p.p["T"]) / _m.sqrt(free)
a = p.acceleration(v, Leader(gap=s_eq, speed=v, actor_id=1, type_id="x"))
print(f"steady following    -> a={a:+.3f} m/s^2  (equilibrium gap={s_eq:.2f} m, expect 0)")
assert abs(a) < 1e-9, a

# 5. closing fast on a stopped leader -> hard braking
a = p.acceleration(8.0, Leader(gap=6.0, speed=0.0, actor_id=1, type_id="x"))
assert a < -3.0, a
print(f"stopped leader, 6 m -> a={a:+.3f} m/s^2  (expect strongly negative)")

# 6. monotonicity: closer leader is never less braking
prev = None
for gap in (40, 30, 20, 12, 8, 5, 3):
    a = p.acceleration(8.0, Leader(gap=float(gap), speed=4.0, actor_id=1, type_id="x"))
    assert prev is None or a <= prev + 1e-9, (gap, a, prev)
    prev = a

# 7. command mapping stays in range
obs = Observation(t=0.0, speed=8.0, x=0.0, y=0.0, heading=0.0,
                  route=[RoutePoint(x=float(i), y=0.0, heading=0.0, s=float(i))
                         for i in range(1, 31)],
                  leader=Leader(gap=4.0, speed=0.0, actor_id=1, type_id="x"))
c = p.act(obs)
assert 0.0 <= c.throttle <= 1.0 and 0.0 <= c.brake <= 1.0 and -1.0 <= c.steer <= 1.0
assert c.brake > 0 and c.throttle == 0
print(f"act() emergency     -> throttle={c.throttle:.2f} brake={c.brake:.2f} steer={c.steer:+.3f}")

# 8. registry + params
assert resolve_policy("idm") is IDMPolicy
assert parse_policy_params(["v0=12.5", "T=1.2"]) == {"v0": 12.5, "T": 1.2}
q = IDMPolicy(v0=12.5)
assert q.p["v0"] == 12.5 and q.p["T"] == IDMPolicy.DEFAULTS["T"]
try:
    resolve_policy("nope")
    raise AssertionError("should have raised")
except ValueError:
    pass
print("registry/params     -> OK")
print("\nALL IDM CHECKS PASSED")
