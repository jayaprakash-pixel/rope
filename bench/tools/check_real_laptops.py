"""P2.5.1: plan vs measured on the real laptops (SD-Turbo 512, one image), before and after the free-RAM correction.

Data: the selftest reports and console results of 2026-11-21 and 2026-11-25 (12 laptops; host and user names removed).
The correction (planner.rs LOW_FREE_GIB / LOW_FREE_FACTOR) multiplies the CPU work by 1.2 when under 2 GiB RAM is free.
Note: the factor was fitted on these same runs (in-sample); new laptops are needed to confirm it.
Run: python bench/tools/check_real_laptops.py   (exit 1 if a claim below no longer holds)
"""
import statistics, sys

LOW_FREE_GIB, LOW_FREE_FACTOR = 2.0, 1.2
# machine, OS, placement, free RAM GiB (None = not in the console output), on battery, plan estimate s, measured s
RUNS = [
    ("HP Victus 16 (Ryzen 5 5600H), dev",   "Windows", "cpu", 1.10, False,  7.0,  8.1),
    ("ASUS Vivobook M6500QC (5600H)",       "Windows", "cpu", None, True,  16.0, 11.3),
    ("Lenovo 82SB (Ryzen 5 6600H)",         "Windows", "cpu", 0.90, False,  9.0, 10.6),
    ("HP 15s (i3-1215U)",                   "Windows", "cpu", None, False,  9.0,  7.9),
    ("HP Victus 15-fb0xxx (5600H)",         "Ubuntu",  "cpu", 2.54, False,  8.0,  7.29),
    ("Acer Aspire A715-76G (i5-12450H)",    "Ubuntu",  "cpu", 2.97, False,  7.0,  6.71),
    ("Acer Predator PHN16-71 (i7-13700HX)", "Ubuntu",  "cpu", 6.21, False,  5.0,  4.10),
    ("Lenovo 81WB (i3-10110U)",             "Windows", "cpu", 1.20, True,  27.0, 30.66),
    ("HP 15s-fq5xxx (i5-1235U)",            "Windows", "cpu", 1.23, False, 17.0, 28.54),
    ("HP Victus 16 GTX 1650, dev",          "Windows", "gpu", 4.0,  False,  5.0,  5.0),
    ("Lenovo 82SB RTX 3050 Laptop",         "Windows", "gpu", 0.90, False,  5.0,  7.1),
    ("ASUS Vivobook RTX 3050 Laptop",       "Windows", "gpu", None, True,   8.0, 13.5),
]

def err(est, meas): return 100.0 * (est - meas) / meas      # + = the plan was cautious, - = the plan was too fast

def main():
    print(f"{'machine':38} {'OS':8} {'on':4} {'free':>5} {'plan':>6} {'real':>6} {'error':>7} {'new plan':>9} {'new error':>10}")
    before, after, low_before, low_after = [], [], [], []
    for name, os_, pl, free, bat, est, meas in RUNS:
        low = free is not None and free < LOW_FREE_GIB
        new = est * LOW_FREE_FACTOR if (low and pl == "cpu") else est
        e0, e1 = err(est, meas), err(new, meas)
        tag = (" battery" if bat else "") + (" low-RAM" if low else "")
        print(f"{name:38} {os_:8} {pl:4} {'?' if free is None else f'{free:.1f}':>5} {est:6.1f} {meas:6.1f} {e0:+6.0f}% {new:9.1f} {e1:+9.0f}%{tag}")
        if pl == "cpu":
            before.append(e0); after.append(e1)
            if low: low_before.append(e0); low_after.append(e1)
    within = lambda xs: sum(abs(x) <= 25 for x in xs)
    print(f"\nCPU runs within 25%: before {within(before)}/{len(before)}, after {within(after)}/{len(after)}")
    print(f"low free RAM (<{LOW_FREE_GIB:.0f} GiB), mean error: before {statistics.mean(low_before):+.0f}%, after {statistics.mean(low_after):+.0f}%")
    print(f"worst too-fast CPU plan: before {min(before):+.0f}%, after {min(after):+.0f}%")
    ok = True
    ok &= all(e < 0 for e in low_before)                         # every low-RAM run was slower than planned
    ok &= all(e > 0 for name, _, pl, free, bat, est, meas in RUNS if pl == "cpu" and free is not None and free >= 2.5 and not bat
              for e in [err(est, meas)])                         # every run with 2.5+ GiB free was faster than planned
    ok &= abs(statistics.mean(low_after)) < abs(statistics.mean(low_before)) / 2
    ok &= within(after) >= within(before) and min(after) > min(before)
    print("OK" if ok else "FAILED: a claim of the correction no longer holds")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
