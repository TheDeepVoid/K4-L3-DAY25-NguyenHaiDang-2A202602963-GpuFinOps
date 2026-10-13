"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def _rightsize_recommend(summary, cat) -> list[dict]:
    """Suggest right-size GPU for each GPU based on MBU, VRAM usage, and bandwidth.

    Checks:
      1. VRAM capacity: target hbm_gb must fit workload peak memory (avoids OOM).
      2. Memory bandwidth: target peak_bw_tbs must satisfy achieved bandwidth.
      3. Cost/GB-VRAM and hourly rate comparison.
    Limitation noted: requires empirical serving benchmark before production rollout
    to guarantee latency SLA is maintained.
    """
    recs = []
    for s in summary:
        cur = s["gpu_type"]
        peak_cur_bw = num(cat[cur]["peak_bw_tbs"])
        hbm_cur = num(cat[cur]["hbm_gb"])
        ondemand_cur = num(cat[cur]["on_demand_hr"])
        mbu_cur = s["mbu"]
        max_mem = s.get("max_mem_gb", 0.0)
        avg_bw = s.get("avg_bw_tbs", 0.0)

        # Target candidate
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        peak_tgt_bw = num(cat[tgt]["peak_bw_tbs"])
        hbm_tgt = num(cat[tgt]["hbm_gb"])
        ondemand_tgt = num(cat[tgt]["on_demand_hr"])

        cost_per_gb_cur = ondemand_cur / hbm_cur if hbm_cur else float("inf")
        cost_per_gb_tgt = ondemand_tgt / hbm_tgt if hbm_tgt else float("inf")
        delta_per_hr = max(0.0, ondemand_cur - ondemand_tgt)
        delta_monthly = delta_per_hr * 24 * 30

        # Feasibility checks
        vram_fits = hbm_tgt >= max_mem
        bw_sufficient = peak_tgt_bw >= avg_bw
        is_lie_or_low_mbu = (s["gpu_util_pct"] >= 90 and s["mfu"] < 0.30) or (mbu_cur < 0.35)

        if cur == tgt:
            status = "optimal"
            reason = f"Đã ở tier thấp nhất ({cur}). Không thể hạ thêm."
            savings = 0.0
            rec_gpu = cur
        elif not vram_fits:
            status = "rejected_vram"
            reason = f"Không thể hạ sang {tgt}: VRAM cần {max_mem:.1f}GB > {tgt} ({hbm_tgt:.0f}GB)."
            savings = 0.0
            rec_gpu = cur
        elif not bw_sufficient:
            status = "rejected_bw"
            reason = f"Không thể hạ sang {tgt}: Cần bandwidth {avg_bw:.2f} TB/s > {tgt} peak ({peak_tgt_bw:.2f} TB/s)."
            savings = 0.0
            rec_gpu = cur
        elif not is_lie_or_low_mbu and mbu_cur >= 0.40:
            status = "retained"
            reason = f"Giữ {cur}: MBU={mbu_cur:.2f} và MFU={s['mfu']:.2f} đang sử dụng tốt tài nguyên."
            savings = 0.0
            rec_gpu = cur
        else:
            status = "recommended"
            reason = (
                f"Đề xuất {cur} -> {tgt}: VRAM đủ ({max_mem:.1f}GB <= {hbm_tgt:.0f}GB), "
                f"BW đủ ({avg_bw:.2f} <= {peak_tgt_bw:.2f} TB/s), "
                f"tiết kiệm ${cost_per_gb_cur:.4f}->${cost_per_gb_tgt:.4f}/GB-hr (${delta_monthly:,.0f}/tháng). "
                f"[Giới hạn: Cần load-test benchmark trước khi áp dụng production]"
            )
            savings = delta_monthly
            rec_gpu = tgt

        recs.append({
            "gpu_id": s["gpu_id"],
            "current": cur,
            "target": rec_gpu,
            "status": status,
            "max_mem_gb": max_mem,
            "hbm_target_gb": hbm_tgt,
            "current_mbu": round(mbu_cur, 3),
            "monthly_savings_usd": round(savings, 2),
            "reason": reason,
        })
    return recs


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0, "mem": [], "bw": []})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        a["mem"].append(num(r.get("mem_used_gb", 0)))
        a["bw"].append(num(r.get("achieved_bw_tbs", 0)))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "max_mem_gb": round(max(a["mem"]), 1) if a["mem"] else 0.0,
            "avg_bw_tbs": round(sum(a["bw"]) / len(a["bw"]), 3) if a["bw"] else 0.0,
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # New: right-sizing recommendations based on MBU
    rightsize_recs = _rightsize_recommend(summary, cat)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")
        print("\n--- Right-Size Recommendations ---")
        for r in rightsize_recs:
            print(f"{r['gpu_id']}: {r['reason']}")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "rightsize_recs": rightsize_recs,
    }


if __name__ == "__main__":
    run()