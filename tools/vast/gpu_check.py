"""Burn-in check of every GPU before paid training (RUNPOD_SETUP.md): bf16 matmul on all GPUs at once while
nvidia-smi samples SM clock, temperature, power, utilization and thermal-slowdown flags; SW/HW thermal-slowdown
counters are diffed from `nvidia-smi -q` before/after. One bad GPU -> exit 1 (return the pod, don't train).
Usage: python gpu_check.py [DUR_S=300] [OUT=/workspace/out/gpu_check]   (needs torch; the training venv has it)
       python gpu_check.py --selftest"""
import json, os, re, statistics as st, subprocess, sys, time

FIELDS = "index,pci.bus_id,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit,utilization.gpu," \
         "clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown"


def burn(dur, out):
    import torch
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
    for _ in range(5): a @ b
    torch.cuda.synchronize(); t0 = time.time(); n = 0
    while time.time() - t0 < dur:
        for _ in range(50): a @ b
        torch.cuda.synchronize(); n += 50
    open(out, "w").write(str(2 * 8192 ** 3 * n / (time.time() - t0) / 1e12))


def counters(txt=None):
    """{bus_id: (sw_us, hw_us)} from the 'Clocks Event Reasons Counters' block; {} if the driver lacks it."""
    txt = txt or subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True).stdout
    res = {}
    for blk in re.split(r"\n(?=GPU [0-9A-Fa-f]{8}:)", txt)[1:]:
        c = blk.split("Counters", 1)[1] if "Counters" in blk else ""
        sw = re.search(r"SW Thermal Slowdown\s*:\s*(\d+)", c); hw = re.search(r"HW Thermal Slowdown\s*:\s*(\d+)", c)
        if sw and hw: res[blk.split("\n", 1)[0][4:].strip().rstrip(":").lower()] = (int(sw.group(1)), int(hw.group(1)))
    return res


def num(s):
    try: return float(s)
    except ValueError: return float("nan")


def verdict(rows, tflops, c0, c1, dur):
    """rows: steady-state nvidia-smi samples (lists of FIELDS strings); tflops {idx: x}; c0/c1 counters()."""
    g = {}
    for r in rows: g.setdefault(int(r[0]), []).append(r)
    clk = {i: st.median(num(r[2]) for r in v) for i, v in g.items()}
    ref_clk, ref_tf = st.median(clk.values()), st.median(tflops.values())
    out = []
    for i, v in sorted(g.items()):
        bus = v[0][1].lower()
        thermal = sum("Active" == r[8].strip() or "Active" == r[9].strip() for r in v) / len(v)
        d = [b - a for a, b in zip(c0.get(bus, (0, 0)), c1.get(bus, (0, 0)))] if bus in c0 and bus in c1 else None
        why = []
        if tflops.get(i, 0) < 0.85 * ref_tf: why.append(f"matmul {tflops.get(i, 0):.0f} < 85% of median {ref_tf:.0f} TFLOPS")
        if clk[i] < 0.85 * ref_clk: why.append(f"SM clock {clk[i]:.0f} < 85% of median {ref_clk:.0f} MHz")
        if thermal > 0.02: why.append(f"thermal slowdown active in {thermal:.0%} of samples")
        if d and max(d) > 0.02 * dur * 1e6: why.append(f"thermal counters +{d[0] / 1e6:.0f}s SW / +{d[1] / 1e6:.0f}s HW")
        util = st.median(num(r[7]) for r in v)
        if util < 90: why.append(f"util {util:.0f}%")
        out.append(dict(gpu=i, bus=v[0][1], tflops=round(tflops.get(i, 0), 1), sm_mhz=clk[i], max_sm_mhz=num(v[0][3]),
                        temp_max=max(num(r[4]) for r in v), power_w=st.median(num(r[5]) for r in v),
                        power_limit_w=num(v[0][6]), util=util, thermal_frac=round(thermal, 3),
                        thermal_counter_delta_us=d, ok=not why, why=why))
    return out


def main(dur, out):
    os.makedirs(out, exist_ok=True)
    n = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
    c0 = counters(); smp = open(f"{out}/samples.csv", "w")
    sampler = subprocess.Popen(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits", "-lms", "2000"], stdout=smp)
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, __file__, "--burn", str(dur), f"{out}/tflops_{i}.txt"],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i in range(n)]
    rc = [p.wait() for p in procs]; sampler.terminate(); smp.close(); c1 = counters()
    lines = open(f"{out}/samples.csv").read().splitlines()
    steady = [[x.strip() for x in l.split(",")] for l in lines[int(len(lines) * 0.3):] if l.count(",") == 9]
    tf = {i: num(open(f"{out}/tflops_{i}.txt").read()) if rc[i] == 0 else 0.0 for i in range(n)}
    res = verdict(steady, tf, c0, c1, time.time() - t0)
    ok = len(res) == n and all(r["ok"] for r in res)
    json.dump(dict(ok=ok, gpus=n, dur_s=dur, at=time.strftime("%F %T %Z"), results=res), open(f"{out}/result.json", "w"), indent=1)
    subprocess.run(f"nvidia-smi -q -d TEMPERATURE,CLOCK,PERFORMANCE,POWER > {out}/nvidia_smi_q_after.txt", shell=True)
    for r in res: print(f"GPU {r['gpu']} {r['bus']} {r['tflops']:6.1f} TFLOPS sm {r['sm_mhz']:.0f}/{r['max_sm_mhz']:.0f} MHz "
                        f"{r['temp_max']:.0f}C {r['power_w']:.0f}/{r['power_limit_w']:.0f} W util {r['util']:.0f}% "
                        f"{'OK' if r['ok'] else 'BAD: ' + '; '.join(r['why'])}")
    print("GPU CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def selftest():
    def row(i, clk, act="Not Active"): return [str(i), f"00000000:{i:02X}:00.0", str(clk), "1980", "60", "690", "700", "100", act, "Not Active"]
    rows = [row(i, 1700) for i in range(8) for _ in range(10)]
    c = {f"00000000:{i:02x}:00.0": (0, 0) for i in range(8)}
    assert all(r["ok"] for r in verdict(rows, {i: 700.0 for i in range(8)}, c, c, 300))
    bad = [row(3, 1000, "Active") if r[0] == "3" else r for r in rows]
    c1 = dict(c); c1["00000000:03:00.0"] = (60_000_000, 0)
    v = verdict(bad, {**{i: 700.0 for i in range(8)}, 3: 300.0}, c, c1, 300)
    assert [r["gpu"] for r in v if not r["ok"]] == [3] and len(v[3]["why"]) == 4, v[3]
    q = "Timestamp : x\n\nGPU 00000000:4B:00.0\n    Clocks Event Reasons\n        SW Thermal Slowdown : Not Active\n" \
        "    Clocks Event Reasons Counters\n        SW Thermal Slowdown : 205200000000 us\n        HW Thermal Slowdown : 1440000000 us\n"
    assert counters(q) == {"00000000:4b:00.0": (205200000000, 1440000000)}, counters(q)
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--burn"]: burn(float(a[1]), a[2])
    elif a[:1] == ["--selftest"]: selftest()
    else: sys.exit(main(float(a[0]) if a else 300, a[1] if len(a) > 1 else "/workspace/out/gpu_check"))
