# 와우마요 R2 미세조정 run 인계 노트 (10/28, vast 8×H100 일본)

## 1. 무인 감시 조건 (`tools/vast/r2ft_autowatch.py`, Mac에서 계속 돎)

Mac에서 `nohup caffeinate -i`로 돌고, 세션과 무관하다. 2분마다 ssh로 상자를 한 번 본다. 이벤트는 같은 내용을 한 번만 디스코드 DM으로 보낸다(🚨 = ALERT, ℹ️ = STAGE). 조건이 풀렸다가 다시 생기면 다시 보낸다. 이벤트가 나도 종료하지 않는다.

- 하트비트: 폴링마다 `~/.r2ft-watch.hb`를 touch한다. **이 파일이 10분 넘게 그대로면 감시가 죽은 것이다.**
- 이벤트 파일: STAGE·ALERT·STATUS를 모두 `~/.r2ft-watch.events`에 `YYYY-MM-DD HH:MM:SS 종류 내용` 한 줄씩 덧붙인다. 메인 세션은 이 파일을 보고 대화에서 전한다.
- 로그: `~/.r2ft-watch.out`
- 상태(보낸 이벤트): `~/.r2ft-watch.57996502.json`
- 시험: `python3 tools/vast/test_r2ft_autowatch.py`(가짜 probe로 조건 40여 개와 STATUS 문구, ssh 없음) 통과. 실제 상자에 읽기 전용 probe를 한 번 돌려 파싱을 확인했고, DM 전송도 확인했다.

**ALERT (🚨)**

| 무엇을 | 조건 |
|---|---|
| 접속 | ssh 3번 연속 실패(6분) |
| vast 상태 | `actual_status`가 running이 아님(10분마다 확인) |
| 체인 | `chain.log`에 FAILED 또는 `pipeline exit` 0 아님 |
| 손절 | 대여 30분 뒤에도 속도 측정·ft step이 하나도 없음 |
| 본 학습 진입 | `R1 step0 sync done` 뒤 15분 안에 ft 손실 줄(`ft/loss_components.jsonl`)이 없음 |
| 프로세스 | 파이프라인 프로세스가 없는데 HALT도 완료도 아님(조용한 죽음) |
| HALT | `$OUT/HALT` 내용 그대로. 관문 불합격이면 "납품본은 B2에 남아 있음"을 붙인다 |
| 중단 스위치 | `ft/ft_stop.json` (general_wo_relabel ADE > s2 +25 %, 2번 연속) |
| 오류 | 가장 최근 단계 로그 끝 300줄에 Traceback, NonFiniteStop, CUDA error, NCCL error, OutOfMemory, Watchdog caught |
| 업로드 | `PUSH_FAILED`, 또는 단계 업로드(b2 push)가 30분 넘게 계속됨 |
| QAD 소형 시험 | `RESULT`가 FAIL 또는 INCOMPLETE(본 학습은 계속) |
| 조용한 정지 | ft·QAD 손실 줄이 15분 동안 늘지 않음. 측정·HF eval step(ft 500 배수, QAD 200 배수)이면 25분, 마지막 step 뒤(최종 측정·저장·관문)면 40분까지 허용. 업로드 중에는 업로드 상한(30분)을 쓴다 |
| QAD 단계 | 탐색(GPU 1장)이 30분 넘음, 탐색이 끝난 뒤 15분 동안 QAD 손실 줄 없음 |
| 납품·관문 | deliver 단계 30분 넘음, qgate 단계 45분 넘음, 단계가 끝난 뒤 15분 동안 다음 단계가 안 시작됨 |
| GPU 놀기 | ft·QAD 학습 중: 8장 **최소** util < 20 %가 5번 연속(10분, rank 하나가 멈춤). 탐색·측정·qgate 중: **최대** util < 20 %가 10분 |
| 느려짐 | ft s/step(최근 3줄 중앙값) > 기대값 × 1.2. 기대값 = 다운로드 중 속도 측정 중앙값(없으면 ft 첫 값) |
| 시간 | 예상 종료가 상자 대여 끝(10:16 KST)의 45분 안 |
| 메트릭 | `metrics_sync.log`의 마지막 `synced`가 30분 넘음 |
| 디스크 | `/workspace` 여유 < 50 GB |
| 자동 정지 | 상자의 `idle_watchdog.sh`가 안 돎 |

**STAGE (ℹ️, 짧게)**
- ft 시작(첫 s/step, 총 step, 예상 끝)
- QAD 시작
- 납품 완료(B2 `runs/r2ft/deliver/final`)
- 관문 결과(통과·불합격, `ratio_vs_s2`)
- 끝

**STATUS (📊, 30분마다 한 줄, 감시를 (다시) 띄울 때도 한 번):**
- 지금 단계: 다운로드 R1 x/190 GB + QAD 소형 시험·속도 측정 / ft 준비·step 0 측정 / ft 학습 / ft 검증 측정 / QAD 탐색 / QAD 학습 / 납품 / 관문 / 끝
- 진행 step / 전체 step, 최근 s/step 중앙값, 예상 끝 시각
- 최근 검증 4종 ADE(일반·동적·정적·곡선)와 중단 기준(`general_wo_relabel`)의 s2 대비 %(한계 +25 %)
- GPU 최소–최대 util, 지금까지 든 비용(vast `dph_total` × 대여 뒤 시간)
- 단계가 바뀌면 STATUS를 기다리지 않고 STAGE DM이 바로 간다.

**끝 (사용자 "다 되면 바로 인스턴스 지우고"):** `PIPELINE_DONE`일 때만 아래를 한다. HALT·중단 스위치로 멈춘 경우는 이어 갈 수 있게 두고 ALERT만 낸다.
1. 상자에서 B2를 확인한다.
   - `runs/r2ft/deliver/final`의 파일 수·바이트 합이 상자의 `deliver/final`과 같은지
   - `deliver/gate.json`, `qgate/summary.json`이 있는지
   - `metrics/h100jp_57996502/`가 비어 있지 않은지
   - 3번 연속 실패하면 지우지 않고 ALERT를 낸다. 상자는 idle watchdog이 약 20분 뒤 멈춘다.
2. 확인되면 `tools/vast/safe_destroy.sh 57996502 /workspace/out/stageR2 b2://morai2026-alpamayo/runs/r2ft/metrics/h100jp_57996502`를 실행한다. 이 스크립트는 상자에서 메트릭을 한 번 더 동기화하고, 성공해야만 삭제한다. 메트릭 경로는 상자가 계속 올리던 `runs/r2ft/metrics/<RUN>`이다. `runs/r2ft` 바로 아래로 동기화하면 단계 폴더와 섞인다.
3. 결과는 DM과 이벤트 파일에 남긴다. 삭제에 성공하면 감시도 끝난다. 실패하면 🚨를 보내고 계속 본다.

**상자 쪽 자동 장치(따로 돎):**
- `idle_watchdog.sh`: 20분 유휴·정체면 상자를 멈춘다(디스크는 유지). 예산 조건은 WARN 로그만 남긴다.
- `metrics_sync.sh --loop 600`: 10분마다 B2 `runs/r2ft/metrics/h100jp_57996502/`로 올리고, ft CSV·그림도 만든다.

## 1b. 중단 스위치 = 횡오차 +100 % (사용자 10/28 04:1x)

**결정 경과**
- 04:1x 사용자: "중단 스위치 끄고 끝까지 학습해. 폐루프로 판단할게."
- 곧이어 사용자: "중단 스위치를 횡방향 ADE 기준으로 바꿔. general_wo_relabel 횡오차가 s2의 2배(+100 %)를 2번 연속 넘으면 멈춤." 이 결정이 앞의 것을 대체한다.
- 새 설정: `stop_ref=general_wo_relabel`, `stop_metric=lat`, `stop_rel_worse=1.0`, `stop_patience=2`. 기준은 step 0의 0.230 m이고, 한계는 0.460 m다.

**배경:** step 500에서 `general_wo_relabel` ADE는 1.334 → 2.058(+54 %)이었다. 증가의 91 %가 달리는 행의 종방향 오차였다. 횡오차는 0.230 → 0.322(+40 %)였다.

**코드**
- `FtSelect(stop_metric=ade|lat|fde|lon)`. 기본은 ade이고, `_stop_hit`가 `metrics[stop_ref][stop_metric]`을 본다.
- `configs/student_ft.yaml`: `stop_metric: lat`, `stop_rel_worse: 1.0`
- `scripts/ft_metrics.py`: 중단선을 그 지표의 그림에 그린다.
- 테스트: `test_stop_switch_lateral`과 설정 테스트 통과. 상자에도 올렸다.

**도는 ft는 메모리에 옛 스위치(ADE +25 %)를 들고 있다.** 상자의 `/workspace/resume_nostop.sh`(로그 `/workspace/resume_nostop.log`)가 처리한다. 재개 값은 `/workspace/resume_params`에 있다. 비어 있으면 8분 기다렸다가 스위치를 끈 채로 잇는다.
- **step 1000에서 옛 스위치가 걸리면:** `ft_stop.json`, HALT, 파이프라인 종료를 본 뒤, 둘을 `$OUT/stopped/`로 옮기고 같은 STAGES·환경에 `FTEXTRA=<resume_params>`를 더해 `pipeline.sh`를 다시 띄운다(`pipeline.log`에 덧붙임). `resume=true`라 checkpoint-1000에서 잇는다.
- **걸리지 않으면:** checkpoint-1000이 저장되고 학습이 1010 넘게 가면, 파이프라인과 학습을 멈춘다. 1000 뒤의 반쯤 쓴 체크포인트는 지우고, checkpoint-1000에서 같은 방식으로 잇는다. 잃는 것은 저장 뒤 수십 step이다. 이렇게 해야 새 스위치가 step 1500부터 판단한다. 옛 스위치는 patience 2라 1000에서 안 걸리면 2000까지 못 건다.

**코드 기준으로 확인한 것**
- `ft()`는 `final/`이 있어도 학습을 다시 돈다. 체크포인트가 있으면 `resume=true`다.
- `done/ft`는 HALT 때 만들어지지 않는다.
- `FtSelect`는 재개 step 뒤의 측정만 버리고, 새 스위치 값을 `ft_select.json`에 덮어쓴다.
- HF는 `TrainerControl`을 복원하지 않는다.

**감시 (`--stop-off`)**
- 옛 스위치(ADE)의 HALT와 `ft_stop.json`은 경보로 보지 않는다. 10분 넘게 재개가 안 되면 ALERT를 낸다.
- **새 스위치(lat)가 걸리면 바로 ALERT를 낸다.** 그때는 자동 재개하지 않는다.
- 재개되면 STAGE "ft 재개"를 한 번 보낸다.
- STATUS는 `ft_select.json`에 있는 지금 run의 스위치를 보여 준다. 재개 뒤에는 "중단 기준 횡오차 s2 대비 +x % (한계 +100 %)"다.

**오탐:** 04:03의 "8.8 s/step", "예상 종료 13:34" ALERT는 오탐이었다. 지금 s/step은 측정·eval·저장 직후 두 줄을 빼고, 최근 5줄의 중앙값으로 잰다.

## 2. 상자

| | |
|---|---|
| vast id | 57996502 (offer 57965573), 8× H100 SXM 80 GB, 일본, Xeon 8480+, 2.15 Gbps, 신뢰도 0.986 |
| ssh | `ssh -p 40073 root@153.126.242.231` (zsh에서는 `bash -c '...'`로 감싸거나 인자를 나눠서 쓴다) |
| 비용 | $19.2/h + 디스크 1 TB 약 $0.19/h. vast 잔액은 대여 시점 $139.93이다. 영상 제공기의 5090(57995833)은 건드리지 않는다 |
| 대여 | 03:12 KST. 호스트가 정한 대여 끝은 **10:16 KST** |
| 인수 확인 | python 1e7 루프 0.93 s, 코어 최고 3.8 GHz, B2 약 150–200 MB/s, `gpu_check` 8장 PASS(675–690 TFLOPS, 700 W) |
| run | `RUN=h100jp_57996502`. 상자 `/workspace/out/stageR2`, 로그는 `/workspace/{stage,chain}.log`, `/workspace/out/pipeline.log`, `$OUT/logs/<단계>.log` |
| B2 | `b2://morai2026-alpamayo/runs/r2ft/{ft,qad,deliver,qgate,done,logs}`, 메트릭 `runs/r2ft/metrics/h100jp_57996502/`, QAD 소형 시험 `runs/r2ft/qadmini/h100jp_57996502/`, gpu_check `runs/r2ft/gpu_check/h100jp_57996502/` |
| 레시피 | 10/28 확정 그대로(하드 타겟, 합성 이력, 중단 +25 %, QAD → 납품 → 관문, `FAST=0`). 바꾸지 않는다 |

**시각 (KST):**
- 03:15 받기 시작
- 03:17 체인 시작
- 03:24 QAD 소형 시험(GPU 0)과 속도 측정(GPU 1–7) 시작
- 속도 측정은 내 버그(`r2ft_speed.sh` 줄 연결)로 바로 죽었다. 고쳐서 03:27에 다시 돌렸고, **2.572 s/step(중앙값, GPU 7장), 최대 메모리 47.2 GB**다. 새 R2 index(121,523행)를 쓰는 것도 로그로 확인했다.
- **QAD 소형 시험: FAIL gate_s2.** 탐색 → 학습 5 step → manifest → 납품(export)은 통과했다(03:24–03:30). `gate_s2`는 행 ADE를 계산한 뒤 `summarize`에서 멈췄다. 원인은 소형 시험 스크립트의 버그다. 체인이 ssh 세션에서 돌아 B2 키가 없었고, start_relabel 파일 다운로드가 조용히 실패했다. 본 파이프라인은 `pipeline.sh`가 PID 1에서 키를 읽고, 재라벨 파일도 `/data/step0/_start_relabel/`에 있어 영향이 없다. `r2ft_qadmini.sh`는 고쳤고, 결과는 손으로 B2 `runs/r2ft/qadmini/h100jp_57996502/`(28파일)에 올렸다. 확인하지 못한 것은 QAD 모델로 도는 관문·기록 세트·요약이다. 이것은 본 run의 qgate에서 처음 돈다. 실패해도 납품은 그 전에 B2에 올라가 있다.
- **QAD 관문 경로 재시험 PASS**(03:42, GPU 0, ft 옆, 51 s): 키를 읽게 고쳐 `gate_s2`·`gate_qad`(QAD fake-quant 모델 로드)·`summary`를 8행으로 다시 돌렸다. ft 진입 뒤 GPU 0 여유 31.5 GB를 확인하고, 단계마다 150 s 상한을 두었다. 8행이라 비율 0.91은 의미가 없다. 결과는 B2 `runs/r2ft/qadmini/h100jp_57996502/qgate_rerun/`. 기록 세트(동적 등)는 다시 돌리지 않았다.
- R1(190 GB)은 03:37:06에 끝났고, 03:37:07에 파이프라인이 시작했다. **ft 시작 03:37:10.**
  - 실측: `total` 104,030, `new_rows_4hz` 31,209, **3,251 step**, `hist_drop_share` 0.342, `hist_synth_share` 0.067
  - step 0 측정 139 s: 일반 1.361, general_wo_relabel 1.334(중단선 1.667), 동적 4.399, 정적 1.463, 곡선 1.548
  - 첫 s/step은 2.436, GPU당 메모리 약 49.5 GB, util 53–81 %
  - 감시의 예상 끝은 07:40 KST다.

**예상:**
- ft는 약 3,342 step × 약 2.5 s ≈ 2.3 h에 측정 약 8번(0.4–0.9 h)이 더해져 약 06:30 KST에 끝난다.
- QAD(탐색 + 800 step)는 약 0.6–0.75 h, 그다음 납품, 관문(약 15분).
- 전체 끝은 약 07:30–08:00 KST, 비용은 약 $85–95다.

## 3. 단계별로 정상일 때 보이는 것

| 단계 | 정상 | 확인 명령 (상자) |
|---|---|---|
| ft | `pipeline.log`에 `=== … ft start`. `logs/ft.log`에 `MixtureDataset: {"total": 약 106,900 …}`, `exclude … 53671 of 2086127`, `synth …: 20886 histories, 847 extra rows`(R2 D)·`14924 histories, 0 extra rows`(기존 복구). step 0 측정 뒤 `ft/loss_components.jsonl`이 10 step마다 한 줄. 500 step마다 `ft_select` 줄(3–7분) | `tail -2 $OUT/ft/loss_components.jsonl; grep ft_select $OUT/logs/ft.log \| tail -1` |
| ft 끝 | 최종 측정 → `ft/final`, `ft_choice.json` → `gate_ft`(NaN 검사) → B2 push(약 15 GB) → `ft done in … h` | `tail -3 /workspace/out/pipeline.log` |
| qad | `qad start` → 탐색(GPU 0만, `qad/search/modelopt_state.pth`) → QAD 800 step(`qad/loss_components.jsonl`, eval 200마다) → `quant_manifest.json` → push(약 30 GB) | `ls $OUT/qad; tail -2 $OUT/qad/loss_components.jsonl` |
| deliver | `deliver/final` + manifest를 B2 `runs/r2ft/deliver`로 올리고 `done/deliver` | `ls $OUT/deliver` |
| qgate | GPU 0에서 관문 2개(ft final, QAD 500행), GPU 1–6에서 기록 세트 → `qgate/summary.json`(`pass`, `ratio_vs_s2`) → `deliver/gate.json`, deliver 다시 push. `ratio_vs_s2` ≥ 1.25면 HALT하지만 납품본은 남는다 | `cat $OUT/qgate/summary.json` |
| 끝 | `pipeline finished`, `$OUT/PIPELINE_DONE` → 감시가 safe_destroy | |

## 4. 흔한 장애와 대응

- **ft 중 조용한 정지·NCCL hang·rank 사망:** `logs/ft.log` 끝과 `nvidia-smi`를 본다. 체크포인트(`ft/checkpoint-<step>`, 500 step마다, 최근 3개 유지)에서 잇는다.
  ```bash
  ssh -p 40073 root@153.126.242.231 'pkill -f scripts/pipeline.sh; pkill -f morai_vla.train; cd /workspace/out/stageR2 && rm -f HALT'
  ssh -p 40073 root@153.126.242.231 'cd /workspace/train && OUT=/workspace/out/stageR2 B2=b2://morai2026-alpamayo/runs/r2ft STAGES="ft qad deliver qgate" QAD_FROM=ft R2ROOT=/data/step0_r2 SEXTRA="model.attn_implementation=flash_attention_2" setsid nohup bash scripts/pipeline.sh >> /workspace/out/pipeline.log 2>&1 < /dev/null &'
  ```
  `resume=true`가 남은 최신 체크포인트를 잡고, `FtSelect`는 그 뒤의 측정을 버린다. 끝난 단계는 `done/` 표시로 건너뛴다.
- **중단 스위치(`ft_stop.json`):** 그 step의 체크포인트가 남고 `gate_ft`가 HALT한다. 레시피 문제일 수 있으므로 사용자에게 알린다. 고치지 않고 이어 가면 안 된다. 버그로 판명되면 train/README 런북 6단계대로 `stopped/`로 옮기고 잇는다.
- **QAD 실패(탐색·학습·manifest):** ft는 이미 B2에 있다(`done/ft`). 고친 뒤 코드를 다시 올리고 `STAGES="qad deliver qgate"`로 위 명령을 다시 띄운다. QAD 학습은 `qad/checkpoint-*`에서 이어진다.
- **qgate 불합격(HALT "QAD gate"):** 납품본(`runs/r2ft/deliver`)과 `deliver/gate.json`(`pass: false`)은 그대로 둔다. 판단은 사용자에게 맡기고, 상자는 safe_destroy한다.
- **B2 push 실패(HALT "B2 push"):** `$OUT/push_<단계>.log`를 본다. 네트워크 문제면 HALT를 지우고 같은 STAGES로 다시 띄운다. 끝난 단계는 건너뛴다.
- **상자 정지(idle watchdog)·vast 상태 이상:** `vastai show instance 57996502`로 본다. 디스크는 유지된다. `vastai start instance 57996502`로 켠 뒤 위처럼 잇는다.
- **감시가 죽음(`~/.r2ft-watch.hb`가 10분 넘게 그대로):** 아래 명령으로 다시 띄운다. 상태 파일 덕분에 보낸 이벤트는 다시 보내지 않는다.
  ```bash
  nohup caffeinate -i python3 tools/vast/r2ft_autowatch.py --id 57996502 --run h100jp_57996502 --rented 1793124757 --end 1793150183 >> ~/.r2ft-watch.out 2>&1 < /dev/null &
  ```
- 크레딧 부족은 상자를 세울 이유가 아니다. 디스코드 DM으로 알린다.

## 5. 끝났을 때

- 감시가 `PIPELINE_DONE`을 보면 safe_destroy를 실행하고 DM한다. 수동으로 할 때:
  ```bash
  bash tools/vast/safe_destroy.sh 57996502 /workspace/out/stageR2 b2://morai2026-alpamayo/runs/r2ft/metrics/h100jp_57996502
  ```
- 확인할 것:
  - B2 `runs/r2ft/done/`에 ft·qad·deliver·qgate 표시가 있다.
  - `runs/r2ft/deliver/final`과 `deliver/gate.json`이 있다.
  - `vastai show instances`에 57996502가 없다.
- 그 뒤 엔진 변환과 폐루프 확인은 사용자 허락을 받은 뒤 한다(winpc).
