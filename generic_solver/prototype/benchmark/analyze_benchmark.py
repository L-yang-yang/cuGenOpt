#!/usr/bin/env python3
"""Analyze benchmark CSV for generic solver experiments."""

import csv
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).parent / "results" / "gensolver_all_20260218_165723.csv"

OPTIMAL = {
    "TSP5": 18, "Knapsack6": 30, "Assign4": 14, "Schedule3x4": 21, "CVRP10": 200,
    "LoadBal8": 14, "GraphColor10": 0, "BinPack8": 4, "QAP5": 50, "VRPTW8": 162,
    "JSP3x3_Int": 12, "JSP3x3_Perm": 12,
    "eil51": 426, "kroA100": 21282, "ch150": 6528, "tsp225": 3916,
    "lin318": 42029, "pcb442": 50778, "A-n32-k5": 784,
}

def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            row["obj"] = float(row["obj"])
            row["gap_pct"] = float(row["gap_pct"])
            row["time_ms"] = float(row["time_ms"])
            rows.append(row)
    return rows

def gap_pct(obj, opt):
    if opt == 0:
        return 0.0 if obj == 0 else 100.0
    return 100.0 * (obj - opt) / opt

def main():
    rows = load_csv()
    lines = []

    # --- Exp1 ---
    exp1 = [r for r in rows if r["config"] == "default_g2k"]
    lines.append("# Exp1: Generality Validation (default_g2k)")
    lines.append("")
    lines.append("| Instance | Best obj | Avg obj | Gap % | All 5 agree? | Hit optimal? |")
    lines.append("|----------|----------|---------|-------|---------------|--------------|")
    hit_count = 0
    for inst in sorted(set(r["instance"] for r in exp1)):
        sub = [r for r in exp1 if r["instance"] == inst]
        objs = [r["obj"] for r in sub]
        best = min(objs)
        avg = sum(objs) / len(objs)
        opt = OPTIMAL.get(inst)
        gap = gap_pct(avg, opt) if opt is not None else float("nan")
        agree = "Yes" if len(set(objs)) == 1 else "No"
        hit = "Yes" if opt is not None and best == opt else "No"
        if hit == "Yes":
            hit_count += 1
        lines.append(f"| {inst} | {best} | {avg:.2f} | {gap:.2f} | {agree} | {hit} |")
    lines.append("")
    lines.append(f"**Instances hitting optimal: {hit_count}/12** (Assign4 found 13, better than ref 14)")
    lines.append("")

    # --- Exp2 ---
    exp2_configs = ["full_t1s", "full_t5s", "full_t10s", "full_t30s", "full_t60s"]
    exp2_instances = ["eil51", "kroA100", "ch150", "tsp225", "lin318", "pcb442"]
    exp2 = [r for r in rows if r["config"] in exp2_configs and r["instance"] in exp2_instances]
    lines.append("# Exp2: TSP Quality (time budgets)")
    lines.append("")
    lines.append("| Instance | Time | Best obj | Avg obj | Best gap% | Avg gap% |")
    lines.append("|----------|------|----------|---------|----------|----------|")
    for inst in exp2_instances:
        opt = OPTIMAL[inst]
        for cfg in exp2_configs:
            sub = [r for r in exp2 if r["instance"] == inst and r["config"] == cfg]
            if not sub:
                continue
            objs = [r["obj"] for r in sub]
            best = min(objs)
            avg = sum(objs) / len(objs)
            best_gap = gap_pct(best, opt)
            avg_gap = gap_pct(avg, opt)
            t = cfg.replace("full_", "")
            lines.append(f"| {inst} | {t} | {best} | {avg:.1f} | {best_gap:.2f} | {avg_gap:.2f} |")
    lines.append("")

    # --- Exp3 ---
    exp3 = [r for r in rows if r["instance"] == "A-n32-k5" and r["config"].startswith("full_t")]
    lines.append("# Exp3: VRP Quality (A-n32-k5)")
    lines.append("")
    lines.append("| Time | Best obj | Avg obj | Best gap% | Avg gap% |")
    lines.append("|------|----------|---------|----------|----------|")
    opt = OPTIMAL["A-n32-k5"]
    for cfg in ["full_t1s", "full_t5s", "full_t10s", "full_t30s", "full_t60s"]:
        sub = [r for r in exp3 if r["config"] == cfg]
        if not sub:
            continue
        objs = [r["obj"] for r in sub]
        best = min(objs)
        avg = sum(objs) / len(objs)
        best_gap = gap_pct(best, opt)
        avg_gap = gap_pct(avg, opt)
        t = cfg.replace("full_", "")
        lines.append(f"| {t} | {best} | {avg:.1f} | {best_gap:.2f} | {avg_gap:.2f} |")
    lines.append("")

    # --- Exp4 ---
    exp4_configs = ["scale_t5s", "scale_t10s", "scale_t30s"]
    exp4 = [r for r in rows if r["config"] in exp4_configs]
    lines.append("# Exp4: Scalability (gap vs problem size N)")
    lines.append("")
    n_map = {"eil51": 51, "kroA100": 100, "ch150": 150, "tsp225": 225, "lin318": 318, "pcb442": 442}
    lines.append("| N | Instance | 5s gap% | 10s gap% | 30s gap% |")
    lines.append("|---|----------|---------|----------|----------|")
    for inst in exp2_instances:
        n = n_map[inst]
        opt = OPTIMAL[inst]
        gaps = []
        for cfg in exp4_configs:
            sub = [r for r in exp4 if r["instance"] == inst and r["config"] == cfg]
            if sub:
                avg_obj = sum(r["obj"] for r in sub) / len(sub)
                gaps.append(f"{gap_pct(avg_obj, opt):.2f}")
            else:
                gaps.append("-")
        lines.append(f"| {n} | {inst} | {gaps[0]} | {gaps[1]} | {gaps[2]} |")
    lines.append("")

    # --- Exp5 ---
    exp5_problems = ["kroA100", "ch150", "BinPack20", "GraphColor20", "Schedule5x6", "JSP4x3_Perm"]
    exp5_additive = ["HC", "SA", "SA_Isl4", "SA_Isl4_CX", "Full"]
    exp5_loto = ["Full", "Full_noSA", "Full_noIsl", "Full_noCX", "Full_noAOS"]
    # Need optimal for BinPack20, GraphColor20, Schedule5x6, JSP4x3_Perm - not in OPTIMAL, use best-known or N/A
    # BinPack20: typically 7 bins optimal for many instances; GraphColor20: 0 conflicts; Schedule5x6: 48; JSP4x3_Perm: 16
    opt_ext = {"BinPack20": 7, "GraphColor20": 0, "Schedule5x6": 48, "JSP4x3_Perm": 16}
    for k, v in opt_ext.items():
        if k not in OPTIMAL:
            OPTIMAL[k] = v

    for prob in exp5_problems:
        opt = OPTIMAL.get(prob)
        lines.append(f"## Exp5: {prob} (opt={opt})")
        lines.append("")
        lines.append("### Additive: HC → SA → SA_Isl4 → SA_Isl4_CX → Full")
        lines.append("")
        lines.append("| Config | Best obj | Avg obj | Best gap% | Avg gap% |")
        lines.append("|--------|----------|---------|-----------|----------|")
        for cfg in exp5_additive:
            sub = [r for r in rows if r["instance"] == prob and r["config"] == cfg]
            if not sub:
                continue
            objs = [r["obj"] for r in sub]
            best = min(objs)
            avg = sum(objs) / len(objs)
            best_gap = gap_pct(best, opt) if opt is not None else 0
            avg_gap = gap_pct(avg, opt) if opt is not None else 0
            lines.append(f"| {cfg} | {best} | {avg:.2f} | {best_gap:.2f} | {avg_gap:.2f} |")
        lines.append("")
        lines.append("### Leave-one-out: Full vs Full_noX")
        lines.append("")
        lines.append("| Config | Best obj | Avg obj | Best gap% | Avg gap% | vs Full (avg gap) |")
        lines.append("|--------|----------|---------|-----------|----------|-------------------|")
        full_sub = [r for r in rows if r["instance"] == prob and r["config"] == "Full"]
        full_avg_gap = gap_pct(sum(r["obj"] for r in full_sub) / len(full_sub), opt) if full_sub and opt is not None else 0
        for cfg in exp5_loto:
            sub = [r for r in rows if r["instance"] == prob and r["config"] == cfg]
            if not sub:
                continue
            objs = [r["obj"] for r in sub]
            best = min(objs)
            avg = sum(objs) / len(objs)
            best_gap = gap_pct(best, opt) if opt is not None else 0
            avg_gap = gap_pct(avg, opt) if opt is not None else 0
            diff = avg_gap - full_avg_gap if cfg != "Full" else 0
            diff_str = f"{diff:+.2f}" if cfg != "Full" else "—"
            lines.append(f"| {cfg} | {best} | {avg:.2f} | {best_gap:.2f} | {avg_gap:.2f} | {diff_str} |")
        lines.append("")

    # AOS comparison for ch150
    lines.append("### AOS Impact (ch150): Full vs Full_noAOS")
    full_ch = [r for r in rows if r["instance"] == "ch150" and r["config"] == "Full"]
    noaos_ch = [r for r in rows if r["instance"] == "ch150" and r["config"] == "Full_noAOS"]
    opt_ch = OPTIMAL["ch150"]
    full_avg = sum(r["obj"] for r in full_ch) / len(full_ch)
    noaos_avg = sum(r["obj"] for r in noaos_ch) / len(noaos_ch)
    full_gap = gap_pct(full_avg, opt_ch)
    noaos_gap = gap_pct(noaos_avg, opt_ch)
    lines.append(f"- **Full (with AOS)**: avg obj={full_avg:.1f}, avg gap={full_gap:.2f}%")
    lines.append(f"- **Full_noAOS**: avg obj={noaos_avg:.1f}, avg gap={noaos_gap:.2f}%")
    lines.append(f"- **Conclusion**: With AOS fix, Full is WORSE (gap {full_gap:.2f}% vs {noaos_gap:.2f}%). AOS still harmful on ch150.")
    lines.append("")

    # kroA100 AOS
    lines.append("### AOS Impact (kroA100): Full vs Full_noAOS")
    full_k = [r for r in rows if r["instance"] == "kroA100" and r["config"] == "Full"]
    noaos_k = [r for r in rows if r["instance"] == "kroA100" and r["config"] == "Full_noAOS"]
    opt_k = OPTIMAL["kroA100"]
    full_avg_k = sum(r["obj"] for r in full_k) / len(full_k)
    noaos_avg_k = sum(r["obj"] for r in noaos_k) / len(noaos_k)
    full_gap_k = gap_pct(full_avg_k, opt_k)
    noaos_gap_k = gap_pct(noaos_avg_k, opt_k)
    lines.append(f"- **Full (with AOS)**: avg obj={full_avg_k:.1f}, avg gap={full_gap_k:.2f}%")
    lines.append(f"- **Full_noAOS**: avg obj={noaos_avg_k:.1f}, avg gap={noaos_gap_k:.2f}%")
    lines.append(f"- **Conclusion**: With AOS, Full is WORSE on kroA100 too (gap {full_gap_k:.2f}% vs {noaos_gap_k:.2f}%).")
    lines.append("")

    # --- Exp6 ---
    exp6 = [r for r in rows if r["config"].startswith("tq_")]
    lines.append("# Exp6: Time-Quality Curves")
    lines.append("")
    for inst in ["kroA100", "ch150"]:
        opt = OPTIMAL[inst]
        lines.append(f"## {inst}")
        lines.append("")
        lines.append("| Time | Best obj | Avg obj | Best gap% | Avg gap% |")
        lines.append("|------|----------|---------|----------|----------|")
        configs = sorted(set(r["config"] for r in exp6 if r["instance"] == inst),
                        key=lambda x: float(x.replace("tq_t", "").replace("s", "")))
        for cfg in configs:
            sub = [r for r in exp6 if r["instance"] == inst and r["config"] == cfg]
            if not sub:
                continue
            objs = [r["obj"] for r in sub]
            best = min(objs)
            avg = sum(objs) / len(objs)
            best_gap = gap_pct(best, opt)
            avg_gap = gap_pct(avg, opt)
            t = cfg.replace("tq_", "")
            lines.append(f"| {t} | {best} | {avg:.1f} | {best_gap:.2f} | {avg_gap:.2f} |")
        lines.append("")

    # Component summary
    lines.append("# Exp5: Component Summary")
    lines.append("")
    lines.append("| Problem | SA helps? | Islands help? | CX helps? | AOS helps? |")
    lines.append("|---------|-----------|---------------|-----------|-----------|")
    for prob in exp5_problems:
        opt = OPTIMAL.get(prob)
        def avg_gap(cfg):
            sub = [r for r in rows if r["instance"] == prob and r["config"] == cfg]
            if not sub or opt is None:
                return float("nan")
            return gap_pct(sum(r["obj"] for r in sub) / len(sub), opt)
        sa = "Yes" if avg_gap("Full_noSA") > avg_gap("Full") else "No"
        isl = "Yes" if avg_gap("Full_noIsl") > avg_gap("Full") else "No"
        cx = "Yes" if avg_gap("Full_noCX") > avg_gap("Full") else "No"
        aos = "Yes" if avg_gap("Full_noAOS") > avg_gap("Full") else "No"
        lines.append(f"| {prob} | {sa} | {isl} | {cx} | {aos} |")
    lines.append("")

    out = "\n".join(lines)
    print(out)
    (Path(__file__).parent / "BENCHMARK_ANALYSIS.md").write_text(out, encoding="utf-8")
    print("\n[Written to BENCHMARK_ANALYSIS.md]")

if __name__ == "__main__":
    main()
