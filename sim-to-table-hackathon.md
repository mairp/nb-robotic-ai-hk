# Sim to Table: NVIDIA × Nebius AI Builders Hackathon

**Berlin · 18 Oct 2026**

Source: [Challenge brief (Google Doc)](https://docs.google.com/document/d/1GCjY5OqIqSCW0tAVaPHSvApo3kKc9BsYThjKmOvqIws/edit)

## The goal

Train a small robot arm (an **SO-101**) to **pick up a 30 mm cube and place it on a green pad**. Judges then run your trained policy on the real arm three times and count how many times it succeeds.

## The problem it's built around

There are two ways to get training demonstrations, and each has a weakness:

- **Simulation (Isaac Sim):** you can generate hundreds of demos quickly, but they look fake. There's one light, one table and one renderer, so a policy trained only on sim usually fails on real hardware.
- **Real teleoperation:** you move a "leader" arm by hand and the "follower" arm copies you, which gives demos that look right. But you can only record a few dozen in an afternoon.

**NVIDIA Cosmos is the fix.** It re-renders the sim videos so they look like your real table, with the same lighting and textures. The motion and geometry stay the same, so the recorded robot actions still match the video. This turns cheap sim data into data that's useful for the real robot.

## The 5-stage pipeline

| # | Step | Tool | Output |
|---|---|---|---|
| 1 | **Simulate**: a scripted expert does pick-and-place many times | Isaac Sim on a Nebius VM | Hundreds of sim episodes |
| 2 | **Transfer**: make sim clips look real. **Reason**: score clips and drop the bad ones | Cosmos Transfer 2.5 + Cosmos 3 Reasoner | Augmented episodes + a curation JSON |
| 3 | **Teleop**: record real demos by hand | Your SO-101 leader/follower pair | A few real episodes |
| 4 | **Train** one policy on sim + Cosmos + real data, **plus** a second one without Cosmos for comparison (an ablation) | Nebius GPU VM, ACT or GR00T | Two checkpoints |
| 5 | **Evaluate**: test both yourself, then judges run the main one | Real arm | Success count |

## Rules that shape the strategy

- **50 real episodes at most.** This stops teams from winning just by recording lots of teleop.
- **300 episodes at most in total.**
- **Your main checkpoint must include sim-derived data**, and you must submit a manifest with episode counts per source.
- **The judges choose where the cube starts.** There are 9 taped grid spots, and judges use a fixed order that's only revealed at eval time. If your policy only learned one starting position, it will fail most trials, so your data needs to cover the whole grid.
- The station (camera, lamp, table) doesn't change between practice and judging.

## Scoring (100 points)

| Criterion | Pts | What earns it |
|---|---|---|
| Accuracy | 30 | 10 per successful trial, 3 trials, 2-minute cap each. Success means the cube is released and resting on the pad |
| Cosmos use | 20 | 10 for Reason: run it on your own videos, save the JSON, and make **one stated curation decision** from it. 10 for Transfer: augment sim data toward the real look |
| Nebius | 20 | Using Nebius Cloud for training |
| Technical competence | 15 | Getting the policy deployed on the real robot |
| Creativity | 15 | "Ensemble of models used" |

## Things worth noticing

- **70 of the 100 points don't depend on the robot succeeding.** Cosmos, Nebius and deployment are almost free points if you do each step and document it. Even a policy that scores 0/3 can still earn a lot.
- **For Transfer, use edge conditioning.** It keeps the structure of the scene fixed, which is what keeps the actions valid. Aim the re-render at the look of your own station.
- **Submission goes through Nebius Object Storage** (`https://storage.<region>.nebius.cloud`). If your folder isn't there by the cutoff, it doesn't count, so upload early.
- **The rules are a bit inconsistent.** The intro says "one ACT policy," but the rules allow ACT or GR00T. Creativity rewards an "ensemble of models" even though you train one main policy; that probably means combining Cosmos Transfer, Cosmos Reason and the policy. It's worth asking an organizer about both.

## Getting started

The brief links to these resources:

- **Starter repo:** `so101-sim-pickplace-gr00t`, which includes the scripted sim expert
- **Guides:** NVIDIA's SO-101 sim-to-real tutorial and the SO-101 calibration guide
- **Nebius Token Factory quickstart**
- **NVIDIA skills:** https://github.com/NVIDIA/skills
