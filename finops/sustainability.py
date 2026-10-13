"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


def carbon_aware_schedule(
    jobs: list[dict],
    catalog: dict,
    days: int = 30,
    baseline_region: str = "us-east-1",
    target_region: str = "europe-north1",
) -> dict:
    """Extension 5: Schedule interruptible workloads to low-carbon/cheaper regions.

    Only interruptible jobs (training, batch eval, sandbox) can be shifted flexibly
    to another region without SLA impact on live user-facing inference.

    Parameters:
      jobs: list of job dicts from workloads.csv
      catalog: price catalog by GPU type (for watts rating)
      days: calculation horizon (30-day month)
      baseline_region: default region (us-east-1, dirty: 380 gCO2/kWh, $0.12/kWh)
      target_region: cleanest region (europe-north1, hydro/clean: 30 gCO2/kWh, $0.09/kWh)

    Returns:
      Dict with per-job carbon before/after, total carbon avoided (kg CO2e),
      and cost trade-off analysis across cleanest vs cheapest regions.
    """
    job_results = []
    tot_baseline_carbon_g = 0.0
    tot_target_carbon_g = 0.0
    tot_baseline_elec_cost = 0.0
    tot_target_elec_cost = 0.0
    tot_energy_kwh = 0.0

    for j in jobs:
        # Check if job is interruptible
        is_interruptible = bool(int(float(j.get("interruptible", 0))))
        if not is_interruptible:
            continue

        gtype = j["gpu_type"]
        ngpu = int(float(j.get("num_gpus", 1)))
        hpd = float(j.get("hours_per_day", 0))
        # Use days parameter (30-day month) for consistent monthly comparison
        gpu_hours = hpd * days * ngpu

        c = catalog.get(gtype, {})
        watts = float(c.get("watts", 500))
        energy_wh = gpu_hours * watts
        energy_kwh = energy_wh / 1000.0
        tot_energy_kwh += energy_kwh

        c_base_g = carbon_g(energy_wh, baseline_region)
        c_tgt_g = carbon_g(energy_wh, target_region)
        cost_base = energy_cost_usd(energy_wh, baseline_region)
        cost_tgt = energy_cost_usd(energy_wh, target_region)

        tot_baseline_carbon_g += c_base_g
        tot_target_carbon_g += c_tgt_g
        tot_baseline_elec_cost += cost_base
        tot_target_elec_cost += cost_tgt

        saved_g = c_base_g - c_tgt_g
        saved_pct = (saved_g / c_base_g * 100.0) if c_base_g > 0 else 0.0

        job_results.append({
            "job_id": j["job_id"],
            "gpu_type": gtype,
            "gpu_hours": round(gpu_hours),
            "energy_kwh": round(energy_kwh, 1),
            "baseline_carbon_kg": round(c_base_g / 1000.0, 2),
            "target_carbon_kg": round(c_tgt_g / 1000.0, 2),
            "carbon_saved_pct": round(saved_pct, 1),
            "baseline_elec_cost": round(cost_base, 2),
            "target_elec_cost": round(cost_tgt, 2),
        })

    total_carbon_saved_kg = (tot_baseline_carbon_g - tot_target_carbon_g) / 1000.0
    total_saved_pct = (
        (tot_baseline_carbon_g - tot_target_carbon_g) / tot_baseline_carbon_g * 100.0
        if tot_baseline_carbon_g > 0 else 0.0
    )

    # Trade-off analysis: cleanest vs cheapest region
    # Cleanest: europe-north1 (30 gCO2/kWh, $0.09/kWh)
    # Cheapest electricity: us-east-wa ($0.055/kWh, 90 gCO2/kWh)
    cheapest_region = min(REGION_PRICE_KWH, key=REGION_PRICE_KWH.get)
    cleanest_region = min(REGION_CARBON, key=REGION_CARBON.get)

    return {
        "interruptible_jobs": job_results,
        "baseline_region": baseline_region,
        "target_region": target_region,
        "cleanest_region": cleanest_region,
        "cheapest_region": cheapest_region,
        "total_energy_kwh": round(tot_energy_kwh, 1),
        "baseline_carbon_kg": round(tot_baseline_carbon_g / 1000.0, 2),
        "target_carbon_kg": round(tot_target_carbon_g / 1000.0, 2),
        "carbon_saved_kg": round(total_carbon_saved_kg, 2),
        "carbon_saved_pct": round(total_saved_pct, 1),
        "baseline_electricity_usd": round(tot_baseline_elec_cost, 2),
        "target_electricity_usd": round(tot_target_elec_cost, 2),
        "electricity_saved_usd": round(tot_baseline_elec_cost - tot_target_elec_cost, 2),
    }
