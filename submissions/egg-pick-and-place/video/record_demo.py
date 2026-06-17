"""
Headless demo / benchmark recorder — FOIB-Egg

Single-episode demo (backward-compatible, no randomisation):
    python video/record_demo.py --out demo.mp4

Multi-episode benchmark:
    python video/record_demo.py --episodes 10 --tier medium --out benchmark.mp4
    python video/record_demo.py --episodes 10 --tier stress --out stress.mp4 --log results.csv

Tiers:   easy | medium (default) | stress
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import mujoco
except ImportError:
    sys.exit("mujoco not found — conda activate robothon")

try:
    import cv2
except ImportError:
    sys.exit("opencv-python not found — pip install opencv-python")

from controller.phase_controller import PhaseController, Phase
from video.overlay import draw

SCENE = os.path.join(os.path.dirname(__file__), "../models/scene.xml")
MAX_STEPS = 20_000

_EGG_DEFAULT  = np.array([0.26,  0.0,  0.800])
_BOWL_DEFAULT = np.array([0.08,  0.22, 0.760])

_HOLD_TERMINAL = 4.0   # s — freeze terminal frame per episode
_HOLD_INTER    = 1.5   # s — inter-episode card
_HOLD_TITLE    = 2.5   # s — opening title card
_HOLD_END      = 4.0   # s — closing results card

# Randomisation ranges per tier.
# Egg Y is capped at ±3 mm across all tiers: the kinematic-attach approach cannot
# tolerate larger Y offsets without fingers pushing the egg before attach fires.
TIER_PARAMS = {
    "easy": dict(
        egg_x   = 0.005,              # ±5 mm  — near-deterministic reach
        egg_y   = 0.001,              # ±1 mm
        egg_rot = np.radians(5),      # ±5°
        bowl_xy = 0.005,              # ±5 mm
    ),
    "medium": dict(
        egg_x   = 0.020,              # ±2 cm  — current validated range
        egg_y   = 0.003,              # ±3 mm
        egg_rot = np.radians(15),     # ±15°
        bowl_xy = 0.015,              # ±1.5 cm
    ),
    "stress": dict(
        egg_x   = 0.030,              # ±3 cm  — near reach limit
        egg_y   = 0.003,              # ±3 mm  — locked (physical constraint)
        egg_rot = np.radians(30),     # ±30°
        bowl_xy = 0.025,              # ±2.5 cm
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="FOIB-Egg benchmark recorder")
    p.add_argument("--out",      default="demo.mp4")
    p.add_argument("--fps",      type=int, default=30)
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=480)
    p.add_argument("--camera",   default="side_cam")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--tier",     default="medium",
                   choices=["easy", "medium", "stress"])
    p.add_argument("--log",      default=None, metavar="PATH",
                   help="write per-episode CSV to PATH")
    return p.parse_args()


# ── scene randomisation ────────────────────────────────────────────────────────

def _randomize(model, data, ctrl, rng, bowl_bid, tp):
    """Perturb bowl body pos (model) and egg freejoint qpos using tier params tp."""
    bdx, bdy = rng.uniform(-tp["bowl_xy"], tp["bowl_xy"], 2)
    model.body_pos[bowl_bid] = _BOWL_DEFAULT + [bdx, bdy, 0.0]

    # reset() → mj_resetData + arm teleport + mj_forward; captures new bowl_pos
    ctrl.reset()

    edx    = rng.uniform(-tp["egg_x"],   tp["egg_x"])
    edy    = rng.uniform(-tp["egg_y"],   tp["egg_y"])
    dtheta = rng.uniform(-tp["egg_rot"], tp["egg_rot"])
    adr = ctrl.egg_qpos_adr
    data.qpos[adr:adr+3]  = _EGG_DEFAULT + [edx, edy, 0.0]
    data.qpos[adr+3:adr+7] = [np.cos(dtheta / 2), 0.0, 0.0, np.sin(dtheta / 2)]
    mujoco.mj_forward(model, data)


# ── title / inter / end cards ─────────────────────────────────────────────────

def _centred_text(img, text, cy, scale, col, thickness=2):
    """Draw horizontally-centred text at row cy. Returns next cy."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cx = max(10, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (cx, cy), font, scale, col, thickness, cv2.LINE_AA)
    return cy + max(24, int(th * 1.9 + 6))


def _title_card(width, height, tier, n_eps, seed):
    """BGR opening card: project name, tier, task summary."""
    card = np.zeros((height, width, 3), dtype=np.uint8)
    cy = height // 2 - 100
    cy = _centred_text(card, "FOIB-Egg",
                       cy, 1.10, (255, 200, 80), thickness=3)
    cy = _centred_text(card, "Fragile Object Integrity Benchmark",
                       cy, 0.62, (200, 200, 200))
    cy += 14
    cy = _centred_text(card,
                       f"Tier: {tier.upper()}   Episodes: {n_eps}   Seed: {seed}",
                       cy, 0.55, (150, 150, 150))
    cy = _centred_text(card,
                       "Shell integrity is the primary scoring criterion",
                       cy, 0.48, (110, 110, 110))
    return card


def _inter_card(width, height, ep_num, n_eps, tier, fail_str, successes):
    """BGR card shown between episodes."""
    card = np.zeros((height, width, 3), dtype=np.uint8)

    if not fail_str:
        shell_text = "SHELL INTEGRITY :  INTACT"
        shell_col  = (80, 200, 80)
    elif fail_str == "OVER-SQUEEZED":
        shell_text = "SHELL INTEGRITY :  OVER-SQUEEZED"
        shell_col  = (60, 60, 220)
    else:
        shell_text = f"SHELL INTEGRITY :  {fail_str}"
        shell_col  = (40, 180, 220)

    cy = height // 2 - 62
    cy = _centred_text(card,
                       f"EPISODE  {ep_num} / {n_eps}   [{tier.upper()}]",
                       cy, 0.78, (210, 210, 210))
    cy = _centred_text(card, shell_text,   cy, 0.70, shell_col)
    cy = _centred_text(card,
                       f"SCORE  {successes} / {ep_num}",
                       cy, 0.65, (155, 155, 155))
    return card


def _end_card(width, height, tier, records):
    """BGR closing card with benchmark statistics."""
    n         = len(records)
    successes = sum(1 for r in records if r["result"] == "SUCCESS")
    over_sq   = sum(1 for r in records if "OVER-SQUEEZED" in r["result"])
    dropped   = sum(1 for r in records if "DROPPED"       in r["result"])
    timeout   = sum(1 for r in records if "TIMEOUT"       in r["result"])
    max_grip  = max(r["peak_grip"] for r in records)
    avg_steps = int(np.mean([r["steps"] for r in records]))
    pct       = 100 * successes // n if n else 0

    card = np.zeros((height, width, 3), dtype=np.uint8)
    cy   = max(28, height // 2 - 148)

    cy = _centred_text(card, "BENCHMARK RESULTS",
                       cy, 0.88, (255, 200, 80), thickness=2)
    cy = _centred_text(card,
                       f"Tier: {tier.upper()}   |   Episodes: {n}",
                       cy, 0.58, (180, 180, 180))
    cy += 6
    # separator
    cv2.line(card, (width // 5, cy), (4 * width // 5, cy), (70, 70, 70), 1)
    cy += 14

    def _row(text, col, scale=0.60):
        nonlocal cy
        cy = _centred_text(card, text, cy, scale, col)

    _row(f"Shell INTACT      {successes:>3} / {n}  ({pct:3}%)",
         (80, 200, 80))
    _row(f"OVER-SQUEEZED     {over_sq:>3} / {n}",
         (60, 60, 220)   if over_sq  else (90, 90, 90))
    _row(f"Dropped           {dropped:>3} / {n}",
         (40, 180, 220)  if dropped  else (90, 90, 90))
    _row(f"Timeout           {timeout:>3} / {n}",
         (150, 150, 220) if timeout  else (90, 90, 90))

    cy += 6
    cv2.line(card, (width // 5, cy), (4 * width // 5, cy), (70, 70, 70), 1)
    cy += 14

    _row(f"Peak grip (max)   {max_grip:.3f} N",  (180, 180, 180), 0.55)
    _row(f"Avg steps / ep    {avg_steps}",        (180, 180, 180), 0.55)
    return card


# ── episode runner ─────────────────────────────────────────────────────────────

def _run_episode(ctrl, model, data, renderer, writer, args,
                 ep_num, n_eps, prior_successes, tier):
    """Run one episode, write frames.

    Returns (phase, fail_str, frames_written, peak_grip).
    peak_grip is sampled at every render frame (every ~17 sim steps).
    """
    render_every   = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))
    frames_written = 0
    phase          = Phase.IDLE
    fail           = None
    peak_grip      = 0.0
    max_contacts   = 0

    for step_i in range(MAX_STEPS):
        phase, fail = ctrl.step()
        mujoco.mj_step(model, data)

        if step_i % render_every == 0:
            renderer.update_scene(data, camera=args.camera)
            rgb  = renderer.render()
            info = ctrl.overlay_info()
            peak_grip   = max(peak_grip,   info["grip_force"])
            max_contacts = max(max_contacts, info.get("contact_count", 0))
            info.update({
                "episode_num":  ep_num,
                "n_episodes":   n_eps,
                "ep_successes": prior_successes,
                "tier":         tier,
            })
            writer.write(draw(rgb, info)[:, :, ::-1])
            frames_written += 1

        if phase in (Phase.DONE, Phase.FAIL):
            updated_successes = prior_successes + (1 if phase == Phase.DONE else 0)
            renderer.update_scene(data, camera=args.camera)
            rgb  = renderer.render()
            info = ctrl.overlay_info()
            peak_grip    = max(peak_grip,    info["grip_force"])
            max_contacts = max(max_contacts, info.get("contact_count", 0))
            info.update({
                "episode_num":  ep_num,
                "n_episodes":   n_eps,
                "ep_successes": updated_successes,
                "tier":         tier,
            })
            hold_bgr = draw(rgb, info)[:, :, ::-1]
            hold_n   = int(_HOLD_TERMINAL * args.fps)
            for _ in range(hold_n):
                writer.write(hold_bgr)
            frames_written += hold_n
            break

    fail_str = fail.value if fail is not None else ""
    return phase, fail_str, frames_written, peak_grip, max_contacts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tp   = TIER_PARAMS[args.tier]

    model    = mujoco.MjModel.from_xml_path(SCENE)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    ctrl     = PhaseController(model, data)
    bowl_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        sys.exit(f"Cannot open video writer for {args.out}")

    # optional CSV log
    csv_fh = csv_writer = None
    if args.log:
        csv_fh     = open(args.log, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["ep", "tier", "result", "grip_peak", "contact_max", "steps"])

    rng          = np.random.default_rng(args.seed)
    n_eps        = args.episodes
    successes    = 0
    total_frames = 0
    records      = []          # accumulated per-episode dicts for end card
    render_every = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))

    print(f"Recording → {args.out}  ({args.fps} fps, camera={args.camera})")
    print(f"  tier={args.tier}  episodes={n_eps}  seed={args.seed}"
          f"  render_every={render_every}")

    t0 = time.time()

    # ── title card (benchmark mode only) ──
    if n_eps > 1:
        title = _title_card(args.width, args.height, args.tier, n_eps, args.seed)
        title_n = int(_HOLD_TITLE * args.fps)
        for _ in range(title_n):
            writer.write(title)
        total_frames += title_n

    # ── episode loop ──
    for ep in range(n_eps):
        ep_num = ep + 1
        print(f"  [EP {ep_num}/{n_eps}]", end=" ", flush=True)

        if n_eps == 1:
            ctrl.reset()                          # demo: fixed positions
        else:
            _randomize(model, data, ctrl, rng, bowl_bid, tp)

        phase, fail_str, nf, peak_grip, contact_max = _run_episode(
            ctrl, model, data, renderer, writer, args,
            ep_num, n_eps, successes, args.tier,
        )
        total_frames += nf

        ep_result = "SUCCESS" if phase == Phase.DONE else f"FAIL:{fail_str}"
        if phase == Phase.DONE:
            successes += 1

        rec = {
            "ep":          ep_num,
            "tier":        args.tier,
            "result":      ep_result,
            "peak_grip":   peak_grip,
            "contact_max": contact_max,
            "steps":       ctrl.step_count,
        }
        records.append(rec)
        if csv_writer:
            csv_writer.writerow([rec["ep"], rec["tier"], rec["result"],
                                  f"{rec['peak_grip']:.4f}", rec["contact_max"],
                                  rec["steps"]])

        print(f"{ep_result}  steps={ctrl.step_count}"
              f"  peak_grip={peak_grip:.3f}N  frames={nf}")

        # inter-episode card (skip after last episode)
        if ep < n_eps - 1:
            card    = _inter_card(args.width, args.height,
                                  ep_num, n_eps, args.tier, fail_str, successes)
            inter_n = int(_HOLD_INTER * args.fps)
            for _ in range(inter_n):
                writer.write(card)
            total_frames += inter_n

    # ── end card (benchmark mode only) ──
    if n_eps > 1:
        end   = _end_card(args.width, args.height, args.tier, records)
        end_n = int(_HOLD_END * args.fps)
        for _ in range(end_n):
            writer.write(end)
        total_frames += end_n

    writer.release()
    renderer.close()
    if csv_fh:
        csv_fh.close()

    elapsed = time.time() - t0
    dur_s   = total_frames / args.fps
    pct     = (100 * successes // n_eps) if n_eps else 0
    print(f"\nDone — {successes}/{n_eps} success ({pct}%)")
    print(f"  total frames: {total_frames}  video: {dur_s:.1f}s"
          f"  wall_time: {elapsed:.1f}s")
    print(f"  Output: {os.path.abspath(args.out)}")
    if args.log:
        print(f"  CSV log: {os.path.abspath(args.log)}")


if __name__ == "__main__":
    main()
