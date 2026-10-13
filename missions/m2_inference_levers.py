"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Cache economics parameters (Extension 3)
    # Assumption: write surcharge is 1.25x base input price (Anthropic caching model);
    # In production document-QA/search, avg prefix reuse is estimated at 4.0 reads.
    AVG_CACHE_READS_ASSUMPTION = 4.0
    CACHE_READ_DISCOUNT = 0.10  # 90% discount on cache hits

    # Compute break-even thresholds per model tier
    be_reads_small = pricing.cache_break_even_reads(
        write_cost_per_m=MODEL_PRICES["small"][0] * 1.25,
        price_in_per_m=MODEL_PRICES["small"][0],
        read_discount=CACHE_READ_DISCOUNT,
    )
    be_reads_large = pricing.cache_break_even_reads(
        write_cost_per_m=MODEL_PRICES["large"][0] * 1.25,
        price_in_per_m=MODEL_PRICES["large"][0],
        read_discount=CACHE_READ_DISCOUNT,
    )

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # OPTIMIZED: cascade (route_tier), prompt caching (with cache_is_worth_it gate), batch API
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]

        # Extension 3 gate: only apply cache discount if reads exceed break-even
        write_cost = pin * 1.25
        worth_it = pricing.cache_is_worth_it(
            avg_cache_reads=AVG_CACHE_READS_ASSUMPTION,
            write_cost_per_m=write_cost,
            price_in_per_m=pin,
            read_discount=CACHE_READ_DISCOUNT,
        )
        effective_cached = cached if worth_it else 0

        opt_cost += pricing.request_cost(
            inp, out, pin, pout, cached_in=effective_cached, batch=is_batch, cache_discount=CACHE_READ_DISCOUNT
        )

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(f"\n--- Extension 3: cache_is_worth_it() ---")
        print(f"Small model break-even reads : {be_reads_small:.2f} reads")
        print(f"Large model break-even reads : {be_reads_large:.2f} reads")
        print(f"Giả định avg_cache_reads={AVG_CACHE_READS_ASSUMPTION:.1f} > break-even => Cache CÓ LỢI (Gate=ACTIVE)")
        # Low reads scenario (1.0 read) comparison
        worth_low = pricing.cache_is_worth_it(1.0, MODEL_PRICES["small"][0] * 1.25, MODEL_PRICES["small"][0])
        print(f"Trường hợp ít reads (avg=1.0 < {be_reads_small:.2f}) => Cache KHÔNG LỢI (Gate=DISABLED, worth_it={worth_low})")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
    }


if __name__ == "__main__":
    run()
