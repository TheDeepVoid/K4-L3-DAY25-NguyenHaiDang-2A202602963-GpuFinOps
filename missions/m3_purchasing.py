"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        elif tier == "reserved_1yr":
            opt_cost = gpu_hours * num(c["reserved_1yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5 — Carbon-aware scheduling for interruptible workloads
    carbon_plan = sustainability.carbon_aware_schedule(jobs, cat, days=DAYS)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"\n--- Extension 5: Carbon-aware Scheduling ---")
        print(f"Dời các job interruptible từ us-east-1 (mặc định) -> europe-north1 (sạch nhất):")
        print(f"{'job_id':18}{'gpu':6}{'energy(kWh)':>12}{'us-east-1(kg)':>15}{'eur-north1(kg)':>16}{'giảm':>8}")
        for cj in carbon_plan["interruptible_jobs"]:
            print(f"{cj['job_id']:18}{cj['gpu_type']:6}{cj['energy_kwh']:>12,.1f}{cj['baseline_carbon_kg']:>15,.1f}{cj['target_carbon_kg']:>16,.1f}{cj['carbon_saved_pct']:>7.1f}%")
        print(f"Tổng phát thải: {carbon_plan['baseline_carbon_kg']:,.1f} kg -> {carbon_plan['target_carbon_kg']:,.1f} kg CO2e "
              f"(-{carbon_plan['carbon_saved_kg']:,.1f} kg, giảm {carbon_plan['carbon_saved_pct']}%)")
        print(f"Chi phí điện  : ${carbon_plan['baseline_electricity_usd']:,.2f} -> ${carbon_plan['target_electricity_usd']:,.2f} "
              f"(-${carbon_plan['electricity_saved_usd']:,.2f})")
        print(f"Trade-off     : Vùng sạch nhất ({carbon_plan['cleanest_region']}, 30g CO2/kWh, $0.09/kWh) "
              f"vừa sạch hơn (-92%) VỪA rẻ hơn điện us-east-1 ($0.12/kWh). "
              f"Vùng rẻ nhất ({carbon_plan['cheapest_region']}, $0.055/kWh, 90g CO2/kWh) rẻ hơn nữa nhưng phát thải gấp 3 lần.")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "carbon_plan": carbon_plan}


if __name__ == "__main__":
    run()
