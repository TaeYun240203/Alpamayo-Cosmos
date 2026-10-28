#!/usr/bin/env python3
"""Unattended watcher of an R2 fine-tune box (user 10/28: "본학습 제대로 들어가는지 QAD에서 사고 안나는지 뭔가 조용히 죽지
않는지 gpu놀지 않는지 ... 인스턴스 비싸단 말이야"). Runs on the Mac under nohup + caffeinate, never exits on an event: every
2 min it probes the box over ssh (one bash script on stdin), decides, sends each ALERT / STAGE once as a Discord DM (bot token
from ~/.claude/channels/discord/.env), touches ~/.r2ft-watch.hb, and keeps going. On PIPELINE_DONE it runs safe_destroy.sh
(metrics sync first) and stops. Conditions (all in decide(); reports/r2ft_run_handoff.md lists them):
  --stop-off (user 10/28: the running ft's ADE +25 % switch is replaced by lateral +100 %): the old switch's ft_stop.json and
         HALT are no alert while /workspace/resume_nostop.sh resumes; ALERT if the HALT stays 10 min; a stop of the new
         (lat) switch is an ALERT; STAGE "ft 재개" once
  ALERT  unreachable 3 polls | vast status not running | chain FAILED | no speed-test or ft step 30 min after rental |
         no ft loss line 15 min after "R1 step0 sync done" | pipeline process gone | HALT (qgate fail: delivery kept) |
         ft stop switch | Traceback/NCCL/OOM/NaN lines | B2 push failed | QAD mini FAIL / INCOMPLETE |
         ft / QAD loss lines stalled 15 min (25 at a measurement/eval step, 40 after the last step) |
         QAD search > 30 min, no QAD loss line 15 min after the search, deliver > 30 min, qgate > 45 min, a push > 30 min,
         nothing between stages > 15 min | min GPU util < 20 % for 10 min while ft / QAD train, max util < 20 % for 10 min
         in search / measurement / qgate | s/step > 1.2 x the speed test's (else the first ft) s/step | ETA within 45 min of the box's end date |
         last metrics sync > 30 min ago | free disk < 50 GB | idle watchdog not running
  STAGE  ft start (first s/step, steps, ETA) | QAD start | delivery done (B2 path) | qgate result | done + destroy result
  STATUS every 30 min (user 10/28 "주기적으로 보고 받고 싶어"): phase, step / steps, s/step, ETA, last 4 val ADEs and the
         stop-switch margin, GPU min-max util, cost so far
Every STAGE / ALERT / STATUS line also goes to ~/.r2ft-watch.events (time-stamped), which the main session reads.
End (user 10/28 "다 되면 바로 인스턴스 지우고"): on PIPELINE_DONE only (never after HALT / stop switch: those stay resumable)
it checks B2 from the box (deliver/final file count + bytes = the box's, deliver/gate.json, qgate/summary.json, metrics),
then safe_destroy.sh; a failed check -> ALERT, no destroy (the box's idle watchdog stops it ~20 min later).
Usage: nohup caffeinate -i python3 tools/vast/r2ft_autowatch.py --id 57996502 --run h100jp_57996502 > ~/.r2ft-watch.out 2>&1 &
Test: python3 tools/vast/test_r2ft_autowatch.py (fake probes, no ssh)."""
import argparse, json, os, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()
B2 = "b2://morai2026-alpamayo/runs/r2ft"
CHAT = "1563731925322174534"
PROBE = r'''
O=/workspace/out/stageR2; P=/workspace/out/pipeline.log; C=/workspace/chain.log
ep() { [ -n "${1// /}" ] && date -d "$1" +%s 2>/dev/null || echo 0; }
echo now=$(date +%s)
pgrep -f "scripts/[p]ipeline.sh" > /dev/null && echo pipe=1 || echo pipe=0
echo r1=$(ep "$(grep -m1 "R1 step0 sync done" $C 2>/dev/null | cut -c1-19)")
echo pstart=$(ep "$(grep -m1 "pipeline start" $C 2>/dev/null | cut -c1-19)")
echo chainfail=$(grep -c "FAILED" $C 2>/dev/null)      # a non-zero pipeline exit shows as HALT / pipe_gone (and a resume is not the chain's)
echo ftstarts=$(grep -cE " ft start$" $P 2>/dev/null)
st=$(grep -E "^=== [0-9-]+ [0-9:]+ [a-z0-9_]+ (start|done in)" $P 2>/dev/null | grep -v "pipeline start" | tail -1)
echo stage=$(echo "$st" | awk '{print $4}'); echo stage_ev=$(echo "$st" | awk '{print $5}'); echo stage_t=$(ep "$(echo "$st" | awk '{print $2" "$3}')")
for s in ft qad; do L=$O/$s/loss_components.jsonl
  echo ${s}_n=$(cat $L 2>/dev/null | grep -c s_per_step); echo ${s}_mt=$(stat -c %Y $L 2>/dev/null || echo 0)
  echo ${s}_step=$(grep s_per_step $L 2>/dev/null | tail -1 | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
  # median of the last 5 train lines, without the 2 lines after each measurement / eval / save step (they carry its minutes)
  echo ${s}_sps=$(python3 -c "import json, statistics, sys; e = int(sys.argv[2]); r = [json.loads(l) for l in open(sys.argv[1]) if 's_per_step' in l][-12:]; v = [x['s_per_step'] for x in r if x['step'] % e not in (0, 10)][-5:]; print(statistics.median(v) if v else '')" $L $([ $s = ft ] && echo 500 || echo 200) 2>/dev/null)
  echo ${s}_sps0=$(grep -o '"s_per_step": [0-9.]*' $L 2>/dev/null | sed -n 3p | awk '{print $2}')
done
echo ft_total=$(grep -m1 -o '"total": [0-9]*' $O/logs/ft.log 2>/dev/null | grep -o '[0-9]*$')
echo r1mb=$(grep -q "pipeline start" $C 2>/dev/null && echo 0 || du -sm /data/step0 2>/dev/null | cut -f1)
echo qmini=$(tail -1 /workspace/out/qadmini.log 2>/dev/null | awk '{print $2}')
echo val=$(python3 -c "
import json; d = json.load(open('$O/ft/ft_select.json')); h = sorted(d['history'], key=lambda x: x['step'])
r, m = d.get('stop_ref', 'general_wo_relabel'), d.get('stop_metric', 'ade')      # the live run's switch (user 10/28: lat +100 %)
g = lambda x: x['ade'].get(r) if m == 'ade' else (x.get('metrics') or {}).get(r, {}).get(m)
print(json.dumps(dict(step=h[-1]['step'], ade=h[-1]['ade'], base=h[0]['ade'], sm=m, sr=d.get('stop_rel_worse'), sv=g(h[-1]), s0=d.get('stop_base') or g(h[0])), separators=(',', ':')))" 2>/dev/null)
echo stopm=$(python3 -c "import json; print(json.load(open('$O/ft/ft_stop.json')).get('stop_metric', 'ade'))" 2>/dev/null)
echo speed_n=$(cat /workspace/out/speed/loss_components.jsonl 2>/dev/null | grep -c s_per_step)
echo speed_sps=$(grep -o '"s_per_step_median": [0-9.]*' /workspace/out/speed/SPEED.json 2>/dev/null | awk '{print $2}')
echo qsearch_mt=$(stat -c %Y $O/qad/search/modelopt_state.pth 2>/dev/null || echo 0)
echo push=$(pgrep -f "b2 sync --threads 8 --exclude-dir-regex" > /dev/null && echo 1 || echo 0)
[ -f $O/HALT ] && echo "halt=$(head -c 200 $O/HALT | tr '\n' ' ')" || echo halt=
echo stop=$([ -f $O/ft/ft_stop.json ] && echo 1 || echo 0); echo done=$([ -f $O/PIPELINE_DONE ] && echo 1 || echo 0)
echo pushfail=$([ -f $O/PUSH_FAILED ] && echo 1 || echo 0)
echo qadmini=$(head -c 80 /workspace/out/qadmini/RESULT 2>/dev/null | tr '\n' ' ')
echo deliver=$([ -f $O/done/deliver ] && echo 1 || echo 0)
echo qpass=$(grep -o '"pass": [a-z]*' $O/qgate/summary.json 2>/dev/null | awk '{print $2}')
echo qratio=$(grep -o '"ratio_vs_s2": [0-9.]*' $O/qgate/summary.json 2>/dev/null | awk '{print $2}')
G=$(ls -t $O/logs/*.log 2>/dev/null | head -1)
echo errlog=$(basename "$G" 2>/dev/null); echo err=$(tail -300 "$G" 2>/dev/null | grep -cE "Traceback|NonFiniteStop|CUDA error|NCCL error|OutOfMemory|Watchdog caught")
echo gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' , | sed 's/,$//')
echo msync=$(ep "$(grep synced /workspace/metrics_sync.log 2>/dev/null | tail -1 | cut -c1-19)")
echo disk=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
echo wd=$(pgrep -f "[i]dle_watchdog.sh" | wc -l)
'''
VERIFY = r'''
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed 's/^/export /')"
O=/workspace/out/stageR2; B=@B2@; R=$(cat /workspace/RUN)
echo deliver_local=$(cd $O/deliver/final 2>/dev/null && find . -type f -printf '%s\n' | awk '{n++; s+=$1} END {print n+0"_"s+0}')
echo deliver_b2=$(b2 ls --recursive --long $B/deliver/final/ | awk '{n++; s+=$5} END {print n+0"_"s+0}')
echo gate_json=$(b2 ls $B/deliver/ | grep -c 'gate.json$')
echo qgate_summary=$(b2 ls $B/qgate/ | grep -c 'summary.json$')
echo metrics=$(b2 ls --recursive $B/metrics/$R/ | wc -l)
'''
STALL, STALL_MEAS, STALL_END = 15 * 60, 25 * 60, 40 * 60      # loss-line silence limits (s)
EVERY = {"ft": 500, "qad": 200}                                 # FtSelect / HF eval interval: a silence at these steps is a measurement
QAD_STEPS = 800


def num(p, k, cast=float, default=None):
    try: return cast(p.get(k) or "")
    except ValueError: return default


def eta(p, now):
    """Epoch at which the whole pipeline should end, or None."""
    st, tot = p.get("stage"), num(p, "ft_total", int)
    if st == "ft" and tot and num(p, "ft_sps") is not None:
        left = (-(-tot // 32) - (num(p, "ft_step", int) or 0)) * num(p, "ft_sps") * 1.25
        return now + left + 1.15 * 3600
    if st == "qad" and num(p, "qad_sps") is not None:
        return now + (QAD_STEPS - (num(p, "qad_step", int) or 0)) * num(p, "qad_sps") * 1.1 + 0.5 * 3600
    if st in ("deliver", "qgate"): return now + 0.4 * 3600
    return None


def phase(p, now):
    st, ev, ft_n, qad_n = p.get("stage"), p.get("stage_ev"), num(p, "ft_n", int, 0), num(p, "qad_n", int, 0)
    if p.get("done") == "1": return "끝"
    if not num(p, "pstart", int, 0):
        return f"다운로드 R1 {num(p, 'r1mb', int, 0) / 1000:.0f}/190 GB + QAD 소형 시험({p.get('qadmini') or p.get('qmini') or '-'})·속도 측정"
    if ev == "done": return f"{st} 끝, 다음 단계로 넘어가는 중"
    if st in ("", None, "ft"):
        if not ft_n: return "ft 준비·step 0 측정"
        return "ft 검증 측정" if num(p, "ft_step", int, 0) % EVERY["ft"] == 0 and now - num(p, "ft_mt", int, 0) > 120 else "ft 학습"
    if st == "qad": return "QAD 탐색" if not num(p, "qsearch_mt", int, 0) else "QAD 학습" if qad_n else "QAD 학습 준비"
    return {"deliver": "납품", "qgate": "관문"}.get(st, st)


def status(p, s, now, cfg):
    """One STATUS line (user 10/28): phase, step / steps, s/step, ETA, val ADEs + stop margin, GPU util range, cost."""
    hm = lambda t: time.strftime("%H:%M", time.localtime(t))
    parts = [hm(now), phase(p, now)]
    if p.get("stage") == "qad" and num(p, "qad_n", int, 0): parts.append(f"{num(p, 'qad_step', int, 0)}/{QAD_STEPS} step, {p.get('qad_sps')} s/step")
    elif num(p, "ft_n", int, 0): parts.append(f"{num(p, 'ft_step', int, 0):,}/{-(-num(p, 'ft_total', int, 0) // 32):,} step, {p.get('ft_sps')} s/step")
    elif p.get("speed_sps"): parts.append(f"속도 측정 {p['speed_sps']} s/step")
    e = eta(p, now)
    if e: parts.append(f"예상 끝 {hm(e)}")
    try: v = json.loads(p.get("val") or "")
    except ValueError: v = None
    if v:
        a, b = v["ade"], v["base"]
        names = [("general", "일반"), ("dynamic", "동적"), ("static", "정적"), ("curve", "곡선")]
        txt = "·".join(f"{k2} {a[k]:.2f}" for k, k2 in names if k in a)
        if v.get("sv") is not None and v.get("s0"):                # the switch the running FtSelect uses (ft_select.json)
            r = v["sv"] / v["s0"] - 1; name = {"ade": "ADE", "lat": "횡오차"}.get(v.get("sm"), v.get("sm"))
            txt += f", 중단 기준 {name} s2 대비 {100 * r:+.1f} %" + (" (스위치 꺼짐)" if v.get("sr") is None else f" (한계 +{100 * v['sr']:.0f} %)")
        parts.append(f"검증(step {v['step']}) {txt}")
    u = [int(x) for x in p.get("gpu", "").split(",") if x.strip().isdigit()]
    if u: parts.append(f"GPU {min(u)}–{max(u)} %")
    if s.get("dph"): parts.append(f"비용 ${(now - cfg['rented']) / 3600 * s['dph']:.1f}")
    return " | ".join(parts)


def decide(p, s, now, cfg):
    """p: probe dict ({} = unreachable), s: persistent state (mutated), -> [(kind, key, text)] currently true (deduped later)."""
    ev = []
    A = lambda key, text: ev.append(("ALERT", key, text))
    S = lambda key, text: ev.append(("STAGE", key, text))
    if not p:
        s["fails"] = s.get("fails", 0) + 1
        if s["fails"] >= 3: A("unreach", "상자에 3번 연속 접속 안 됨")
        return ev
    s["fails"] = 0
    age = now - cfg["rented"]
    if s.get("vast") not in (None, "running"): A("vast", f"vast 상태가 running이 아님: {s['vast']}")
    if num(p, "chainfail", int): A("chainfail", "체인 FAILED (chain.log)")
    ft_n, qad_n = num(p, "ft_n", int, 0), num(p, "qad_n", int, 0)
    if age > 1800 and not num(p, "speed_n", int) and not ft_n: A("stoploss", "대여 30분이 지나도 속도 측정·본 학습 step 없음 (손절 기준)")
    r1 = num(p, "r1", int, 0)
    if r1 and not ft_n and now - r1 > 15 * 60: A("ft_nostart", f"R1 받기 끝({time.strftime('%H:%M', time.localtime(r1))}) 뒤 15분 동안 ft 손실 줄 없음")
    qm = p.get("qadmini", "").strip()
    if qm and not qm.startswith("PASS"): A("qadmini", f"QAD 소형 시험 {qm} (B2 {B2}/qadmini/) — 본 학습은 계속")
    halt = p.get("halt", "").strip()
    # user 10/28: the old ADE switch's halt is resumed by resume_nostop.sh on the box (new switch: lateral +100 %); a stop of the
    # new switch (ft_stop.json stop_metric lat) is a real ALERT
    stop_halt = cfg.get("stop_off") and "ft stop switch" in halt and p.get("stopm", "ade") in ("", "ade")
    s["stophalt_since"] = (s.get("stophalt_since") or now) if stop_halt else None
    if stop_halt and now - s["stophalt_since"] > 600: A("resume_late", "중단 스위치 HALT 뒤 10분이 지나도 자동 재개가 안 됨 (/workspace/resume_nostop.log)")
    if halt and not stop_halt: A("halt", "HALT: " + halt + (" — 납품본은 B2에 남아 있음" if "QAD gate" in halt else ""))
    if p.get("stop") == "1" and not stop_halt and not (cfg.get("stop_off") and p.get("stopm", "ade") in ("", "ade")):
        A("stop", f"ft 중단 스위치 ({p.get('stopm') or 'ade'} 기준, general_wo_relabel이 s2 대비 한계를 2번 연속 넘음)")
    if num(p, "err", int): A("err:" + p.get("errlog", ""), f"오류 줄 {p['err']}개: logs/{p.get('errlog')}")
    if p.get("pushfail") == "1": A("pushfail", "B2 업로드 실패 (PUSH_FAILED)")
    done = p.get("done") == "1"
    if num(p, "pstart", int, 0) and p.get("pipe") == "0" and not done and not halt: A("pipe_gone", "파이프라인 프로세스가 없음 (HALT·완료 아님)")
    if 0 < num(p, "disk", int, 999) < 50: A("disk", f"디스크 여유 {p['disk']} GB")
    ms = num(p, "msync", int, 0)
    if age > 1800 and now - (ms or cfg["rented"]) > 1800: A("msync", "metrics_sync 마지막 성공이 30분 넘음")
    if p.get("wd") == "0": A("wd", "상자의 idle watchdog이 안 돎 (자동 정지 없음)")

    st, ev_kind, st_t = p.get("stage"), p.get("stage_ev"), num(p, "stage_t", int, 0)
    utils = [int(x) for x in p.get("gpu", "").split(",") if x.strip().isdigit()]
    push = p.get("push") == "1"
    s["push_since"] = (s.get("push_since") or now) if push else None
    if push and now - s["push_since"] > 1800: A("push_long", "B2 업로드가 30분 넘게 계속됨")
    util_rule = None                                             # "min" | "max" | None for this poll
    if done: pass
    elif ev_kind == "done" and not push and now - st_t > 15 * 60 and p.get("pipe") == "1": A("between", f"{st} 끝난 뒤 15분 동안 다음 단계가 시작되지 않음")
    elif ev_kind == "start" and st in ("ft", "qad") and not push:
        n, mt, step = (ft_n, num(p, "ft_mt", int, 0), num(p, "ft_step", int, 0)) if st == "ft" else (qad_n, num(p, "qad_mt", int, 0), num(p, "qad_step", int, 0))
        last = (-(-num(p, "ft_total", int, 10 ** 9) // 32)) if st == "ft" else QAD_STEPS
        if st == "qad" and not n:
            qs = num(p, "qsearch_mt", int, 0)
            if not qs and now - st_t > 30 * 60: A("qsearch_long", "QAD 탐색이 30분 넘음")
            if qs and now - qs > 15 * 60: A("qad_nostart", "QAD 탐색 뒤 15분 동안 QAD 손실 줄 없음")
            util_rule = "max" if not qs else None
        elif n:
            meas = step % EVERY[st] == 0
            lim = STALL_END if step >= last - 10 else STALL_MEAS if meas else STALL
            if now - mt > lim: A("stall:" + st, f"{st} 손실 줄이 {int((now - mt) / 60)}분째 그대로 (step {step}) — 조용한 정지 의심")
            util_rule = None if step >= last - 10 else "max" if meas or now - mt > 300 else "min"
    elif ev_kind == "start" and st == "deliver" and now - st_t > 1800: A("deliver_long", "납품 단계가 30분 넘음")
    elif ev_kind == "start" and st == "qgate":
        if now - st_t > 2700: A("qgate_long", "관문 단계가 45분 넘음")
        util_rule = "max"
    if util_rule and utils:
        low = (min(utils) if util_rule == "min" else max(utils)) < 20
        s["low"] = s.get("low", 0) + 1 if low else 0
        if s["low"] >= 5: A("util", f"GPU util {'최소' if util_rule == 'min' else '최대'} 20 % 미만 10분 ({p['gpu']}), 단계 {st}")
    else: s["low"] = 0
    sps, sps0 = num(p, "ft_sps"), num(p, "ft_sps0")
    exp = num(p, "speed_sps") or sps0                            # expected: the download-time speed test (same per-GPU work), else the first ft value
    if st == "ft" and sps and exp and sps > 1.2 * exp: A("slow", f"ft {sps} s/step > 기대 {exp}의 1.2배")
    e = eta(p, now)
    if e and cfg.get("end") and e > cfg["end"] - 2700:
        A("eta", f"예상 종료 {time.strftime('%H:%M', time.localtime(e))}가 상자 대여 끝 {time.strftime('%H:%M', time.localtime(cfg['end']))} 45분 안")

    if ft_n >= 3 and sps0:
        steps = -(-num(p, "ft_total", int, 0) // 32)
        S("ft_start", f"ft 시작: {sps0} s/step, {steps} step, 예상 끝 {time.strftime('%H:%M', time.localtime(e)) if e else '?'}")
    if num(p, "ftstarts", int, 0) >= 2 and ft_n and not p.get("halt"):
        S("ft_resumed", f"ft 재개: 새 중단 스위치(횡오차 +100 %)로 checkpoint에서 이어 감 (step {num(p, 'ft_step', int, 0)}, {p.get('ft_sps')} s/step)")
    if st == "qad" or qad_n: S("qad_start", "QAD 시작")
    if p.get("deliver") == "1": S("deliver", f"납품 완료: {B2}/deliver/final")
    if p.get("qpass"): S("qgate", f"관문 {'통과' if p['qpass'] == 'true' else '불합격 (납품본은 유지)'}: ratio_vs_s2 {p.get('qratio')}")
    if done: S("done", "PIPELINE_DONE")
    return ev


def probe(cfg):
    try:
        r = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
                            "-p", cfg["port"], cfg["host"], "bash -s"], input=PROBE, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired: return {}
    return dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l and l.split("=", 1)[0].isidentifier())


def dm(text):
    tok = next((l.split("=", 1)[1].strip() for l in (HOME / ".claude/channels/discord/.env").read_text().splitlines()
                if l.startswith("DISCORD_BOT_TOKEN=")), "")
    req = urllib.request.Request(f"https://discord.com/api/v10/channels/{CHAT}/messages", method="POST",
                                 data=json.dumps({"content": text[:2000], "allowed_mentions": {"parse": []}}).encode(),
                                 headers={"Authorization": f"Bot {tok}", "Content-Type": "application/json", "User-Agent": "r2ft-watch"})
    try: return urllib.request.urlopen(req, timeout=20).status == 200
    except Exception as x: print("dm failed:", x, flush=True); return False


def vast(cfg, key):
    try:
        d = json.loads(subprocess.run(["vastai", "show", "instance", cfg["id"], "--raw"], capture_output=True, text=True, timeout=60).stdout)
        return d.get(key)
    except Exception: return None


def log_event(kind, text):
    with open(HOME / ".r2ft-watch.events", "a") as f: f.write(f"{time.strftime('%F %T')} {kind} {text}\n")
    print(time.strftime("%T"), kind, text, flush=True)


def verify(cfg):
    """B2 holds the delivery (file count + bytes = the box's), deliver/gate.json, qgate/summary.json and metrics -> (ok, facts)."""
    try:
        r = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", "-p", cfg["port"], cfg["host"], "bash -s"],
                           input=VERIFY.replace("@B2@", B2), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired: return False, {}
    v = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l and l.split("=", 1)[0].isidentifier())
    ok = (v.get("deliver_local", "0_0") == v.get("deliver_b2") and not v.get("deliver_local", "0_0").startswith("0_")
          and num(v, "gate_json", int, 0) > 0 and num(v, "qgate_summary", int, 0) > 0 and num(v, "metrics", int, 0) > 0)
    return ok, v


def main():
    global B2
    a = argparse.ArgumentParser()
    a.add_argument("--id", required=True); a.add_argument("--run", required=True); a.add_argument("--rented", type=int)
    a.add_argument("--end", type=int); a.add_argument("--every", type=int, default=120); a.add_argument("--no-destroy", action="store_true")
    a.add_argument("--b2", default=B2, help="the run's B2 prefix (10/28 stop round: b2://morai2026-alpamayo/runs/r2stop)")
    a.add_argument("--stop-off", action="store_true", help="user 10/28: the ft stop switch is off (box resumes); its HALT is not an alert")
    a = a.parse_args(); B2 = a.b2
    url = subprocess.run(["vastai", "ssh-url", a.id], capture_output=True, text=True).stdout.strip()   # ssh://root@host:port
    hp = url.removeprefix("ssh://")
    sf = HOME / f".r2ft-watch.{a.id}.json"
    s = json.loads(sf.read_text()) if sf.exists() else {"sent": {}}
    cfg = dict(id=a.id, host=hp.rsplit(":", 1)[0], port=hp.rsplit(":", 1)[1], rented=a.rented or s.get("rented") or int(time.time()),
               end=a.end or vast({"id": a.id}, "end_date") or None, stop_off=a.stop_off)
    s["rented"] = cfg["rented"]
    print(f"{time.strftime('%T')} watch {a.id} {hp} rented {time.strftime('%H:%M', time.localtime(cfg['rented']))}"
          f" end {time.strftime('%H:%M', time.localtime(cfg['end'])) if cfg['end'] else '?'}", flush=True)
    k = 0
    while True:
        (HOME / ".r2ft-watch.hb").touch()
        now = time.time()
        if k % 5 == 0:
            s["vast"] = vast(cfg, "actual_status") or s.get("vast")
            s["dph"] = vast(cfg, "dph_total") or s.get("dph")
        p = probe(cfg); k += 1
        evs = decide(p, s, now, cfg)
        active = {key for _, key, _ in evs}
        logged = s.setdefault("logged", [])
        for kind, key, text in evs:
            if key in s["sent"]: continue
            if key not in logged: log_event(kind, text); logged.append(key)
            if dm(f"[r2ft {a.run}] {'🚨' if kind == 'ALERT' else 'ℹ️'} {text}"): s["sent"][key] = int(now)
        for key in list(s["sent"]) + logged:                      # an alert that cleared may fire again later; STAGE keys stay
            if key not in active and not key.startswith(("ft_start", "ft_resumed", "qad_start", "deliver", "qgate", "done")):
                s["sent"].pop(key, None); logged.remove(key) if key in logged else None
        if p and now - s.get("status_t", 0) >= 1800:              # STATUS every 30 min (and at a (re)start)
            txt = status(p, s, now, cfg); log_event("STATUS", txt)
            if dm(f"[r2ft {a.run}] 📊 {txt}"): s["status_t"] = now
        sf.write_text(json.dumps(s))
        if p.get("done") == "1" and not s.get("verify_failed"):   # PIPELINE_DONE only (HALT / stop switch stay resumable)
            if a.no_destroy: print("done; --no-destroy", flush=True); return
            ok, v = verify(cfg)
            if not ok:
                s["verify_tries"] = s.get("verify_tries", 0) + 1
                if s["verify_tries"] >= 3:                        # no destroy: the box's idle watchdog stops it ~20 min after the pipeline
                    s["verify_failed"] = True; t = f"끝났지만 B2 확인 실패, 상자를 지우지 않음 (idle watchdog이 약 20분 뒤 정지): {v}"
                    log_event("ALERT", t); dm(f"[r2ft {a.run}] 🚨 {t}")
                sf.write_text(json.dumps(s)); time.sleep(a.every); continue
            r = subprocess.run(["bash", str(ROOT / "tools/vast/safe_destroy.sh"), a.id, "/workspace/out/stageR2", f"{B2}/metrics/{a.run}"],
                               capture_output=True, text=True, timeout=3600)
            ok = r.returncode == 0
            print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
            t = (f"끝: B2 확인(납품 {v.get('deliver_b2')} 파일_바이트, gate.json, qgate, 메트릭 {v.get('metrics')}개) 뒤 상자 삭제 완료"
                 if ok else "끝났지만 safe_destroy 실패 — 상자가 아직 과금 중, 확인 필요") + f" (B2 {B2}/deliver)"
            log_event("STAGE" if ok else "ALERT", t); dm(f"[r2ft {a.run}] {'ℹ️' if ok else '🚨'} {t}")
            if ok: return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
