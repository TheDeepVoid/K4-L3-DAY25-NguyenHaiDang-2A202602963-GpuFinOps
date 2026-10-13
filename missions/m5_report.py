"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability with reasoning breakdown ---
    # Get token counts from M2 data
    total_tokens = r2["total_tokens"]
    reasoning_tokens = r2.get("reasoning_tokens", 0)
    non_reasoning_tokens = r2.get("non_reasoning_tokens", 0)

    # If M2 didn't pass reasoning tokens, estimate from is_reasoning column
    if reasoning_tokens == 0 or non_reasoning_tokens == 0:
        # Re-count from token_usage.csv
        import csv
        rows = []
        with open(os.path.join(ROOT, "data", "token_usage.csv"), newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        rt = sum((int(r["input_tokens"]) + int(r["output_tokens"])) for r in rows if int(r.get("is_reasoning", 0)))
        nrt = total_tokens - rt
        reasoning_tokens = rt
        non_reasoning_tokens = nrt

    wh_reasoning = sustainability.wh_per_query(reasoning_tokens, is_reasoning=True)
    wh_non_reasoning = sustainability.wh_per_query(non_reasoning_tokens, is_reasoning=False)
    carbon_reasoning = sustainability.carbon_g(wh_reasoning)
    carbon_non_reasoning = sustainability.carbon_g(wh_non_reasoning)
    energy_cost_reasoning = sustainability.energy_cost_usd(wh_reasoning)
    energy_cost_non_reasoning = sustainability.energy_cost_usd(wh_non_reasoning)

    # --- sustainability summary ---
    sust = {
        "wh_per_query": round((wh_reasoning + wh_non_reasoning) / max(1, reasoning_tokens + non_reasoning_tokens) * 1e6, 2),
        "carbon_g": round(carbon_reasoning + carbon_non_reasoning, 3),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
        "reasoning_wh": round(wh_reasoning, 2),
        "non_reasoning_wh": round(wh_non_reasoning, 2),
        "reasoning_carbon": round(carbon_reasoning, 3),
        "non_reasoning_carbon": round(carbon_non_reasoning, 3),
        "reasoning_energy_cost": round(energy_cost_reasoning, 4),
        "non_reasoning_energy_cost": round(energy_cost_non_reasoning, 4),
        "reasoning_pct_tokens": round(reasoning_tokens / total_tokens * 100, 1) if total_tokens else 0.0,
        "reasoning_pct_traffic": round(len([r for r in rows if int(r.get("is_reasoning", 0)) == 1]) / len(rows) * 100, 1) if rows else 8.4,
    }

    md = report.build_report(baseline, optimized, levers, sustainability=sust)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
