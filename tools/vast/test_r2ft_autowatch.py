"""Fake-probe checks of r2ft_autowatch.decide (no ssh, no Discord): python3 tools/vast/test_r2ft_autowatch.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from r2ft_autowatch import decide, phase, status   # noqa: E402

T0 = 1_800_000_000
CFG = dict(rented=T0, end=T0 + 9 * 3600)
BASE = dict(now=str(T0), pipe="0", r1="0", pstart="0", chainfail="0", stage="", stage_ev="", stage_t="0", ft_n="0", ft_mt="0",
            ft_step="", ft_sps="", ft_sps0="", qad_n="0", qad_mt="0", qad_step="", qad_sps="", qad_sps0="", ft_total="", speed_n="0",
            speed_sps="", qsearch_mt="0", push="0", halt="", stop="0", done="0", pushfail="0", qadmini="", deliver="0", qpass="",
            qratio="", errlog="ft.log", err="0", gpu="90,90,90,90,90,90,90,90", msync=str(T0), disk="800", wd="1")


def keys(now, s=None, cfg=CFG, **kw):
    s = {} if s is None else s
    return {k for _, k, _ in decide(BASE | {"msync": str(now - 60)} | kw, s, now, cfg)}


def ft_running(now, **kw):
    return dict(pipe="1", r1=str(T0 + 1500), pstart=str(T0 + 1500), stage="ft", stage_ev="start", stage_t=str(T0 + 1500), ft_n="40",
                ft_mt=str(now - 60), ft_step="400", ft_sps="2.5", ft_sps0="2.4", speed_sps="2.4", ft_total="106900") | kw


now = T0 + 600
assert keys(now) == set(), keys(now)                                                    # staging, all fine
assert "stoploss" in keys(T0 + 1900)                                                     # 30 min, no speed / ft step
assert "stoploss" not in keys(T0 + 1900, speed_n="5")
assert "ft_nostart" in keys(T0 + 1500 + 16 * 60, r1=str(T0 + 1500), pstart=str(T0 + 1500), pipe="1", speed_n="5")   # rule 1
assert "ft_nostart" not in keys(T0 + 1500 + 14 * 60, r1=str(T0 + 1500), speed_n="5")
s = {}
for _ in range(2): assert "unreach" not in {k for _, k, _ in decide({}, s, now, CFG)}
assert "unreach" in {k for _, k, _ in decide({}, s, now, CFG)}                           # 3rd failed probe
assert "vast" in keys(now, {"vast": "exited"})
now = T0 + 3 * 3600
assert keys(now, **ft_running(now)) == {"ft_start"}, keys(now, **ft_running(now))       # healthy ft: only the STAGE event
assert "stall:ft" in keys(now, **ft_running(now, ft_mt=str(now - 16 * 60)))              # rule 2: silent hang
assert "stall:ft" not in keys(now, **ft_running(now, ft_mt=str(now - 16 * 60), ft_step="500"))   # measurement step: 25 min
assert "stall:ft" in keys(now, **ft_running(now, ft_mt=str(now - 26 * 60), ft_step="500"))
assert "stall:ft" not in keys(now, **ft_running(now, ft_mt=str(now - 30 * 60), ft_step="3341"))  # after the last step: 40 min
assert "stall:ft" not in keys(now, **ft_running(now, ft_mt=str(now - 50 * 60), push="1"))  # pushing: the push cap applies
s = {"push_since": now - 31 * 60}; assert "push_long" in keys(now, s, **ft_running(now, push="1"))
s = {}                                                                                  # rule 3: one rank idle (min util) 5 polls
for _ in range(4): assert "util" not in keys(now, s, **ft_running(now, gpu="90,90,90,5,90,90,90,90"))
assert "util" in keys(now, s, **ft_running(now, gpu="90,90,90,5,90,90,90,90"))
s = {}                                                                                  # measurement: max util rule, one idle GPU is fine
for _ in range(6): assert "util" not in keys(now, s, **ft_running(now, ft_step="1000", ft_mt=str(now - 400), gpu="90,0,0,0,0,0,0,0"))
assert "slow" in keys(now, **ft_running(now, ft_sps="3.0"))                              # 3.0 > 1.2 x 2.4
assert "eta" in keys(now, **ft_running(now, ft_step="10", ft_sps="4.0"))                 # ends after end - 45 min
assert "err:ft.log" in keys(now, **ft_running(now, err="2"))
assert "pipe_gone" in keys(now, **ft_running(now, pipe="0"))
assert "pipe_gone" not in keys(now, **ft_running(now, pipe="0", halt="2026 ft stop"))
assert {"halt", "stop"} <= keys(now, **ft_running(now, pipe="0", halt="x", stop="1"))
assert "msync" in {k for _, k, _ in decide(BASE | ft_running(now, msync=str(now - 31 * 60)), {}, now, CFG)}
assert "disk" in keys(now, **ft_running(now, disk="40")) and "wd" in keys(now, **ft_running(now, wd="0"))
assert "qadmini" in keys(now, qadmini="FAIL train") and "qadmini" not in keys(now, qadmini="PASS 812")
q = dict(pipe="1", stage="qad", stage_ev="start", stage_t=str(now - 31 * 60), ft_n="300")
assert {"qsearch_long", "qad_start"} <= keys(now, **q)                                   # rule 4: search cap
assert "qad_nostart" in keys(now, **q | dict(qsearch_mt=str(now - 16 * 60)))
assert "stall:qad" in keys(now, **q | dict(qsearch_mt=str(now - 40 * 60), qad_n="20", qad_step="210", qad_mt=str(now - 16 * 60)))
assert "stall:qad" not in keys(now, **q | dict(qsearch_mt=str(now - 40 * 60), qad_n="20", qad_step="200", qad_mt=str(now - 16 * 60)))
assert "deliver_long" in keys(now, pipe="1", stage="deliver", stage_ev="start", stage_t=str(now - 31 * 60))
assert "qgate_long" in keys(now, pipe="1", stage="qgate", stage_ev="start", stage_t=str(now - 46 * 60))
assert "between" in keys(now, pipe="1", stage="qad", stage_ev="done", stage_t=str(now - 16 * 60))
assert {"deliver", "qgate"} <= keys(now, deliver="1", qpass="false", qratio="1.31", halt="QAD gate: 1.31")
assert "done" in keys(now, done="1", deliver="1", qpass="true")
# STATUS line (user 10/28): phase, progress, s/step, ETA, val ADEs + stop margin, GPU range, cost
assert phase(BASE | dict(r1mb="135000", qmini="train"), now).startswith("다운로드 R1 135/190 GB + QAD 소형 시험(train)")
assert phase(BASE | ft_running(now, ft_step="1000", ft_mt=str(now - 300)), now) == "ft 검증 측정"
assert phase(BASE | ft_running(now), now) == "ft 학습" and phase(BASE | ft_running(now, ft_n="0"), now) == "ft 준비·step 0 측정"
assert phase(BASE | dict(pstart="1", stage="qad", stage_ev="start"), now) == "QAD 탐색"
val = '{"step":1000,"ade":{"general":1.5,"general_wo_relabel":1.49,"dynamic":2.1,"static":1.3,"curve":1.8},"base":{"general_wo_relabel":1.40},"sm":"ade","sr":0.25,"sv":1.49,"s0":1.40}'
t = status(BASE | ft_running(now, val=val), {"dph": 19.4}, now, CFG)
assert all(x in t for x in ("ft 학습", "400/3,341 step", "2.5 s/step", "예상 끝", "일반 1.50·동적 2.10·정적 1.30·곡선 1.80",
                            "ADE s2 대비 +6.4 % (한계 +25 %)", "GPU 90–90 %", "비용 $58.2")), t
# --stop-off (user 10/28): the stop-switch HALT is the box's to resume; alert only when it is not resumed in 10 min
OFF = CFG | {"stop_off": True}
sh = ft_running(now, pipe="0", halt="2026-10-28 04:28 ft stop switch at step 1000", stop="1", ft_step="1000")
s = {}; assert {"halt", "stop", "pipe_gone"}.isdisjoint(keys(now, s, OFF, **sh)), keys(now, {}, OFF, **sh)
assert "resume_late" in keys(now + 601, s, OFF, **sh)
assert {"halt", "stop"} <= keys(now, {}, CFG, **sh)                                     # switch on: alerts as before
assert "halt" in keys(now, {}, OFF, **ft_running(now, pipe="0", halt="QAD gate: 1.31"))  # other HALTs still alert
assert "ft_resumed" in keys(now, {}, OFF, **ft_running(now, ftstarts="2"))
lv = '{"step":1500,"ade":{"general":1.5},"base":{},"sm":"lat","sr":1.0,"sv":0.322,"s0":0.230}'
assert "중단 기준 횡오차 s2 대비 +40.0 % (한계 +100 %)" in status(BASE | ft_running(now, val=lv), {}, now, OFF)
lat_stop = sh | dict(halt="2026 ft stop switch at step 2000", stopm="lat")                  # the new switch fired: real ALERT
assert {"halt", "stop"} <= keys(now, {}, OFF, **lat_stop), keys(now, {}, OFF, **lat_stop)
assert "gpufail" in keys(now, {}, CFG, gpufail="1", pstart="0")                        # staging gpu_check FAIL: destroy (10/29)
assert "gpufail" not in keys(now, {}, CFG, **ft_running(now, gpufail="1"))               # once training runs it is no reason
# braking onset (10/29): popout_onset brake + general / static false_brake with step 0's; silent when the run has none
assert "제동" not in t
bv = val[:-1] + ',"bk":{"popout_onset":{"brake":0.42},"popout_pre":{"brake":0.08},"general":{"brake":0.1,"false_brake":0.03},"static":{"false_brake":0.05}},' \
     '"bk0":{"popout_onset":{"brake":0.31},"popout_pre":{"brake":0.06},"general":{"false_brake":0.02}}}'
t = status(BASE | ft_running(now, val=bv), {}, now, CFG)
assert "제동 onset 42 % / pre 8 % (s0 31 % / pre 6 %), 오제동 일반 0.03 (s0 0.02)·정적 0.05 |" in t, t
assert "오제동 일반" in status(BASE | ft_running(now, val=val[:-1] + ',"bk":{"general":{"false_brake":0.03}},"bk0":{}}'), {}, now, CFG)
print("test_r2ft_autowatch: all passed")
