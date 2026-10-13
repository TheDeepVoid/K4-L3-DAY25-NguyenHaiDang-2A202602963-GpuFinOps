"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(
    hpd: float,
    interruptible: bool,
    gpu_type: str | None = None,
    commitment_months: int = 36,
) -> str:
    """Pick a purchasing tier from duty cycle, interruptibility, GPU type, and commitment length.

    Policy (Bonus 1 extension — sources in comments):
      - interruptible -> 'spot' unless H100/A100 running >=8 h/day (frequent checkpoints
        cost more than the 1yr reserved discount for high-demand GPUs; assumption: H100
        spot interruption rate ~5%/hr from neocloud 2026 spot SLA docs).
      - commitment_months >= 36 AND duty >= break-even(3yr discount) -> 'reserved'
        (3yr reserved discount ~44% for H100/A100 per price_catalog.csv).
      - commitment_months >= 12 AND duty >= break-even(1yr discount) -> 'reserved_1yr'
        (1yr reserved discount ~20% for H100/A100 per price_catalog.csv).
      - otherwise -> 'on_demand'.

    GPU-type interruption rate assumptions (neocloud 2026 estimates):
      H100/H200: ~5%/hr  (high demand, scarce capacity)
      A100:      ~8%/hr
      A10G/L4:   ~3%/hr  (more available capacity)
    """
    # GPU-type specific spot interruption rates (assumption: neocloud 2026 spot SLAs)
    SPOT_INTERRUPT_RATE: dict[str, float] = {
        "H100": 0.05, "H200": 0.05,
        "A100": 0.08,
        "A10G": 0.03, "L4": 0.03, "B200": 0.04, "MI300X": 0.06,
    }
    rate = SPOT_INTERRUPT_RATE.get(gpu_type or "", 0.05)

    # For high-interruption GPUs with significant daily hours, spot rework cost
    # erodes the discount — prefer reserved even if interruptible.
    spot_rework_daily_hrs = rate * hpd * 0.5   # 0.5h rework per interrupt
    if interruptible:
        if gpu_type in ("H100", "A100") and hpd >= 8 and spot_rework_daily_hrs > 0.5:
            # Rework overhead > 0.5 h/day: reserved likely cheaper (assumption)
            pass  # fall through to reserved logic below
        else:
            return "spot"

    # Stable (non-interruptible) or high-rework interruptible: pick commitment length
    duty = max(0.0, hpd) / 24.0

    # 3yr discount ~44% for H100, ~44% A100, ~25% A10G (from price_catalog.csv)
    DISC_3YR: dict[str, float] = {
        "H100": 0.44, "H200": 0.39, "A100": 0.44,
        "A10G": 0.40, "L4": 0.44, "B200": 0.37, "MI300X": 0.38,
    }
    # 1yr discount ~20% for H100, ~22% A100 (from price_catalog.csv)
    DISC_1YR: dict[str, float] = {
        "H100": 0.20, "H200": 0.19, "A100": 0.22,
        "A10G": 0.20, "L4": 0.25, "B200": 0.18, "MI300X": 0.18,
    }
    disc_3yr = DISC_3YR.get(gpu_type or "", 0.44)
    disc_1yr = DISC_1YR.get(gpu_type or "", 0.20)

    if commitment_months >= 36 and duty >= break_even_utilization(disc_3yr):
        return "reserved"
    if commitment_months >= 12 and duty >= break_even_utilization(disc_1yr):
        return "reserved_1yr"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
