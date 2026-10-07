# Task 2: stacking two cubes — two different ways to drop something

Same harness as task 1 (pick-and-place), same controller setup: I look at a
rendered frame plus object/gripper telemetry, decide the next move, execute,
look again. This time the task needed two full pick-place cycles plus a
precision constraint — cube B has to land centered on top of cube A, not just
"near" a pad.

It did not finish. That's the useful part.

## Failure 1: the fling (fixed)

Picking up cube A went cleanly — same as task 1. Transporting it did not.
I moved it in one shot: a single large IK target roughly 25cm away in x and
32cm in y, held with a grip that was firm but not maximal
(`gripper_closure: 0.88`, i.e. resisted, not clamped to zero). The resulting
swing was fast enough that cube A came free mid-flight and landed **1.2
meters** from the arm's base — more than 4x the arm's ~0.67m reach, gone for
good. 

Diagnosis: the transport primitive had no notion of speed or smoothness. One
target, one big move, physics fills in whatever velocity that implies. Fine
for a firm grip and a short hop; not fine for a marginal grip and a long one.

Fix: break transport into 2–3 shorter waypoints. Retried the whole episode
from scratch (the harness is fully seeded, so this was a genuine redo, not a
fudge) — cube A made it to the target pad intact this time. 

## Failure 2: death by a thousand nudges (not fixed)

Cube A placed. Cube B picked up cleanly. Then, aligning B over A for the
stack, cube A started drifting — a little further each time I brought the
gripper down near it: 

| pass | cube A xy | drift from previous |
|---|---|---|
| placed | (0.575, 0.188) | — |
| align attempt 1 | (0.534, 0.245) | 7.0 cm |
| align attempt 2 | (0.519, 0.279) | 3.9 cm |
| align attempt 3 | (0.506, 0.306) | 3.7 cm |
| pulled straight up to reset | (0.517, 0.369) — **on the floor** | knocked clean off the table edge |

Each individual nudge was small enough that I read it as "close enough, just
retarget" rather than "something is colliding." The gripper's fingers extend
further than I was accounting for, and every approach clipped cube A's near
edge before B was actually positioned above it. No single correction was the
mistake — the mistake was treating each nudge as independent instead of
noticing the trend. Four small pushes in the same direction is not four
unrelated events. 

This is the sharper version of "recovery attempts sometimes became more
falling": the first failure (the fling) was a single bad action with an
obvious, immediate signal. The second was a slow accumulation across several
actions that each looked individually fine, with a verifier (`cube_b_to_
cube_a_dist_xy`) that only measures the current gap, not the trend in the
target's own position. Fixing failure 1 did nothing for failure 2 — they're
different bugs wearing the same symptom. 

Cube A ended up almost exactly at the edge of the arm's reach (0.64m against
a ~0.67m max) — not clearly recoverable, and precisely the kind of
near-the-limit position where a grasp attempt is most likely to fail again.
I stopped there rather than force it. 

## What I'd change before retrying

- **Track the target's position across attempts, not just the current
  error.** A verifier that flags "this thing has moved 3 times in the same
  direction" would have caught this after nudge 2, not after the object was
  on the floor.

- **Approach stacking from directly above, at a safer clearance height,**
  rather than sweeping in at an angle — the nudges were almost certainly
  fingertip-to-cube contact during descent, not during the deliberate
  grasp/place motions.
  
- **Separate "adjust because I'm imprecise" from "adjust because the world
  changed."** I kept doing the former when the latter was happening.

None of this needed a bigger model. It needed the loop to check a fact it
already had access to (cube A's position, logged at every step) instead of
only checking the fact it was trying to optimize (distance to a moving
target).

---
*Run artifacts: `contact_sheet_task2.png` (12 key frames across both failure
modes), `state/log.jsonl` (full step-by-step telemetry for this episode).*
