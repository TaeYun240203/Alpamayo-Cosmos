# 와우마요 DAgger 러너 (`train/dagger/`)

액터가 실시간으로 달린 판단과 쉐도 라벨을 계속 받아 LoRA를 학습한다. 몇 분마다 LoRA를 공개하고, 액터는 다음 에피소드부터 새 LoRA로 달린다. 베이스는 납품물(QAD `final/`, NVFP4/FP8 가짜 양자화)이고 얼려 둔다. `train/rl` 형식의 LoRA만 학습한다.

| 파일 | 내용 |
|---|---|
| `runner.py` | 러너 본체: DDP, 계획기, 학습 step, 공개 전 검사, 공개(+B2), 메트릭, 재개 |
| `spool.py` | spool 읽기: 준비된 에피소드, 4 Hz 행 고르기, 라벨 검사·결합, 학습 샘플 만들기 |
| `loss.py` | DAgger 행 손실(fm + traj_ce, valid 마스크), KV를 지키는 LLM 체크포인팅 |
| `history.py` | 공개 버전마다 이력 의존도(zero·swap 이동량) 기록 |
| `pools.py` | (Mac) replay 풀: 신호 감속·접근 풀, 정지 유지 풀, 미래 녹색 제외 행, R1 일반 replay 제외 목록 → `assets/dagger/` |
| `../configs/dagger.yaml` | 배치 몫(DAgger·replay 소스), 행 태그·가중치, 기록용 val 세트 |
| `fake.py` | 가짜 spool + 가짜 라벨(실제 카메라·BEV 크기, critic_obs, 신호 필드) |
| `tiny_test.sh` | CPU 한 바퀴: 대기 → 첫 갱신 → 공개 → 재개, gloo 2프로세스 |
| `dryrun.sh` | 학습 상자에서 R1을 받는 동안 돌리는 dry run (20 step, 샘플/s, 메모리, 공개 1회) |
| `run.sh` | 본 실행: spool 내려받기 + 메트릭 올리기 + 러너 |
| `brev_up.sh`, `brev_stage.sh` | (Mac → Brev 상자) 생성 → 코드·b2 계정 → 인수 확인·받기·8장 dry run (사용자 10/28 "Brev H100") |
| `../tests/test_dagger.py` | 단위 테스트 14개 |

`rl/policy.py`의 `LoRA`에는 `targets` 인자만 덧붙였다. 기본값은 예전 그대로 expert이고, 나머지 파일은 고치지 않았다.

## 입력

- **액터 기록**: eval 브랜치 `rl_episode.py` 머리말 형식이다. `<spool>/<actor>/epK/steps.jsonl`(판단마다 `t_sim`, `req`, `out`), `K.req.bin`/`K.out.bin`, `done.json`.
- **쉐도 라벨**: `epK/dagger_labels.npz`
  - `step` int32 (N,): steps.jsonl 줄 번호
  - `ak` float32 (N,64,2): 물리 단위 (a m/s², κ 1/m), 0.1 s 간격
  - `valid` bool (N,64)
  - `source` int8 (N,): 0 계획기, 1 공식 경로
  - 선택 `signal` int8 (N,): t0에 그 행이 따르는 신호(`_ego` 코드: 0 없음, 1 적, 2 황, 3 녹)
  - 선택 `stop_d` float32 (N,): 앞축에서 다음 판정 정지선까지 거리(m, 없으면 inf). 이 두 필드는 dagger-shadow에 요청해 두었다. 없으면 신호 태그가 걸리지 않는다.
  - 다른 세션 파일이라 읽을 때 모양·dtype·step 범위를 검사한다. 값이 유한하지 않은 점은 invalid로 바꾼다. 형식이 틀린 에피소드는 `bad_episode` 이벤트로 남기고 건너뛰며, 러너는 멈추지 않는다.
- **준비 조건**: done.json, 라벨, 고른 행의 req·out 파일이 모두 있어야 한다. b2 sync는 `.b2.sync.tmp`로 받은 뒤 이름을 바꾸므로, 다 받기 전에는 읽지 않는다.

## 행과 손실

- **4 Hz 행** (사용자 10/28): 판단마다 가장 가까운 0.25 s 격자점을 구한다(시뮬 시간 기준). 격자점마다 가장 가까운 판단 하나를 행으로 쓴다. 판단이 0.25 s보다 늦어 비는 격자점에는 행이 없다.
- **버리지 않는다**: 라벨이 없는 4 Hz 행과 valid 점이 하나도 없는 행은 학습할 수 없다. 이런 행은 `spool.jsonl`에 `unlabeled`, `no_valid`로 센다.
- **배치** = `--batch` 샘플(전 rank 합). 몫은 `configs/dagger.yaml`에서 배치 전체 기준으로 정한다. `dagger.share`와 replay 소스 몫의 합은 1이어야 하고, 러너가 확인한다. 바꿀 때는 `--override dagger.replay.sources.<소스>.share=…`를 쓴다.

| 몫 (기본값) | 내용 | 상태 |
|---|---|---|
| DAgger 0.50 | 쉐도 라벨 행 | 제안 (첫 브리핑의 50 %) |
| `r1_general` 0.25 | s2 학습 행(셀 균형). 출발 재라벨 행과 미래 녹색 행은 뺀다. 손실 fm + traj_ce + lm | 제안 |
| `signal_approach` 0.10 | R1 학습 행 중 적신호(`_ego` signal 1)에서 감속하거나, 아직 움직이며(v ≥ 0.5) 판정 정지선이 60 m 안인 행. 미래 녹색 행은 뺀다. 정답 손실(fm + traj_ce) | 확정 (사용자 10/28) |
| `signal_hold` 0.05 | 적신호에서 이미 서 있는 행("stays stopped"). 미래 녹색 행은 뺀다. 정답 손실 | 확정 (사용자 10/28) |
| `new_a` 0.05 / `new_b` 0.025 / `new_d` 0.025 | R2 A/B/D (R2 ft와 같은 2:1:1). A·B는 동적 물체 때문에 실제로 급제동하거나 선 행만 쓴다(아래) | 몫은 제안 (합 10 %), A·B 행 고르기는 확정 (사용자 10/28) |
| `start_relabel`, `old_recovery`, `old_curve` 0 | 쓸 수 있게 넣어 두었다. 몫이 0이면 읽지 않는다 | — |

  - 부모 세션이 전한 제안(DAgger 50, R1 일반 30, 신호 10, R2 10)에 신호 15 %를 넣으면 합이 105가 된다. 그래서 기본값은 R1 일반을 25 %로 줄였다. 확정이 아니다.
  - **적신호 풀** (`pools.py`): 움직임 태그는 `scripts/ft_regression.py`와 같다. v0 = v[i], dv3 = v[i+30] − v0, v64 = v[i+64]이다. "stays stopped"는 v0 < 0.5이고 v64 < 0.5다. 그 밖에 dv3 > 1이면 "accel", dv3 < −1이면 "brake"다. 표본 3,000행에서 ft_regression과 태그가 100 % 같았다.
    - 감속·접근: 신호 1이면서 "brake"인 행, 또는 "stays stopped"가 아니고 v0 ≥ 0.5이며 다음 판정 정지선(앞축 기준, 5 m 뒤까지 셈)이 60 m 안인 행
    - 정지 유지: 신호 1이면서 "stays stopped"인 행
    - 두 풀은 겹치지 않는다.
  - **미래 녹색 제외** (사용자 10/28): 기준은 `relabel_stage1.py`의 signal_change·signal_unknown과 같다.
    - 다음 판정 정지선(앞축 기준, 5 m 뒤까지 셈)이 정답이 닿는 거리 + 50 m 안에 있을 때만 본다.
    - 6.4 s 창 `signal[i..i+64]`에서 적·황 뒤에 녹색이 나오면 `to_green`, 0(GT 없음)이 있으면 `unknown`으로 뺀다.
    - R1 소스에 건다. `r1_general`, `old_recovery`, `old_curve`는 `exclude`로 걸려 있어, 끌 때는 그 줄을 지운다.
    - `signal_approach`, `signal_hold`는 `pools.py`가 풀을 만들 때 이미 뺐다.
    - R2에는 걸지 않는다. R2 `_ego` 신호는 0과 3뿐이라, 녹색 규칙에 걸릴 행이 없고 0 규칙은 행만 지운다.
    - 이 기준대로 적·황 → 녹만 뺀다. 녹 → 황·적 변화는 남긴다.
  - **R2 A·B 동적 풀** (`pools.py r2`, 사용자 10/28 "실제로 급제동하는 경우 위주로. 아예정지하지 않는 경우는 제외"):
    - 사건은 R2 주행기의 AEB 제동 구간과 양보(yield on..off) 구간이다(`_meta/r2_episodes.parquet`).
    - 그중 차가 실제로 느려진 사건만 쓴다. 기준은 시작 3 s 전부터 끝까지 어느 시점에서든 3 s 뒤 속도가 2 m/s 이상 떨어졌거나, 구간 중 또는 끝난 뒤 3 s 안에 v < 0.5인 경우다.
    - 행은 τ가 [시작 − 6.4 s, 끝 + 3 s] 안인 것이다. 6.4 s 정답이 그 사건과 겹치거나, 물체가 지나간 뒤 다시 출발하는 구간이다.
    - 움직이는 물체(actor_kph > 0)가 있는 에피소드만 쓴다.
    - `--glitch net` 인덱스에서 새로 살아난 행 중 6.4 s 안에 위 기준으로 급제동하는 행도 넣는다. 급가속만 있는 행은 넣지 않는다.
    - 결과(학습 행, B2 net 인덱스 기준):

| | 에피소드 | 학습 행 | 2 Hz | 살린 행 포함 |
|---|---|---|---|---|
| A | 396 중 308 | 57,617 중 27,498 (48 %) | 5,518 | 5,109 중 4,219 |
| B | 160 중 85 (움직이는 물체 109) | 20,852 중 5,955 (29 %) | 1,204 | 2,140 중 1,482 |

  - **보이지 않던 물체 제외** (`pools.py r2 --r2logs` → `r2_unseen.parquet`, new_a·new_b의 `exclude`, 사용자 10/28 미래 정보 누출):
    - R2 bag에는 GT 물체 토픽이 없다(Object Info 미녹화). 그래서 물체 위치를 다시 만든다.
      - 사건은 r2.log의 `brake object k`와 `yield on k:종류 .. off`다. 위의 "실제로 느려진 사건" 기준을 같이 쓴다.
      - 위치는 장면 매니페스트의 출발(s, lat) → 도착(s_end, lat_end) 직선 위에서, r2.log 2 s마다의 `disp`(이동 거리)만큼 간 점으로 잡는다. 다른 물체는 매니페스트 위치에 둔다.
      - 판정은 공통 함수 `tools/scorer/label_check.visible()`(쉐도·채점기와 같음)을 쓴다. LiDAR BEV(뒤 20, 앞 60, 좌우 20 m) 안이고, 차량이나 2 m 이상 물체에 가리지 않으면 보인다.
      - 사건 시작 − 6.4 s부터 그 물체가 처음 보이는 격자 시각 전까지의 행을 뺀다.
    - r2.log 물체 번호 → 매니페스트 물체는 종류(에셋 5xxxxxxx 보행자, 나머지 장애물)와 종류 안의 순서로 맞춘다. 종류나 움직임이 안 맞는 에피소드(A·B 556개 중 59개)는 판정 불가로 두고 남긴다.
    - 크기는 R2에 기록이 없어 종류별 기본값이다(차 모양 소품·차 4.6×1.9, 사람 0.6×0.6, 기타 1.0×0.6 m).
    - 결과(학습 행):
      - A: 27,498 중 6,819행 제외(24.8 %). 판정 불가 2,966, 사건 없이 급제동 규칙으로만 들어온 행 1,890은 남긴다(전 split 기준).
      - B: 5,955 중 800행 제외(13.4 %). 판정 불가 508, 사건 없음 278은 남긴다.
    - 제외 대부분은 t0에 물체가 BEV 60 m 밖이던 경우다. 가림 때문인 경우는 적다. 카메라는 더 멀리 볼 수 있지만, 공통 함수 기준(BEV)을 따랐다.
    - R1은 물체 기록이 없어 하지 않았다. R1 replay 소스(r1_general, 신호 풀)는 이 제외가 없다.
  - replay는 MixtureDataset 고정 schedule 순서로 읽는다. 손실은 R2 ft의 `StudentVLA.forward` 그대로다(gt 소스는 history dropout 0.5).
  - **DAgger**: 아직 학습하지 않은 행을 오래된 것부터 먼저 쓴다. 그래서 모든 행이 적어도 한 번 쓰인다. 새 행이 모자라면 이미 쓴 행(집계 데이터)에서 무작위로 채운다. 이때도 몫(50 %)은 그대로다.
    - 행별 학습 횟수를 센다(사용자 10/28, 과적합 확인). step마다 `uses_step_mean`·`uses_step_max`(이번 step 행), `uses_mean`·`uses_max`(지금까지 학습한 모든 행), `rows_used_2plus`·`rows_used_5plus`를 기록하고 `state.pt`에도 둔다.
- **DAgger 행 가중치** (`dagger.weights`, 사용자 10/28 확정 ×2, 기본으로 켬): 행의 fm과 그 행 토큰의 CE에 곱한다. 평균은 행·토큰 수로 낸다(가중치 합으로 나누면 micro 1에서 가중치가 사라진다). 두 태그가 다 붙으면 큰 값을 쓴다.
  - 신호 접근: 라벨 `signal`이 1·2이고 `stop_d` ≤ 60 m, 또는 signal 1이고 v < 0.5(적신호 정지). v는 critic_obs 0번이다.
  - 차선 이탈: 라벨 `source` 1, 또는 critic_obs `lane_offset`(7번, 차로 중심 기준 좌 +)의 절댓값 ≥ 0.5 m.
  - 기준값은 `dagger.tags`에 있다. step마다 `tag_signal_rows`, `tag_lane_rows`, `dagger_signal/*`, `dagger_lane/*`를 기록한다.
- **DAgger 손실** (`loss.py`): 라벨 (a, κ)를 모델의 정규화 통계로 바꾼다.
  - fm: valid 점만 평균한다. 행마다 평균하고 다시 행끼리 평균한다.
  - traj_ce: 라벨의 이산 토큰(2t + d)을 teacher forcing으로 넣는다. CE는 valid 점과 시작·끝 토큰에만 건다.
  - lm과 history dropout은 쓰지 않는다. 액터의 실제 이력을 그대로 쓴다.
  - 라벨이 GT 미래일 때 R2 ft의 GT 행 손실과 같음을 테스트로 확인했다.
- **늦음**: 게이트를 두지 않는다. 행마다 주행 당시의 LoRA 버전을 기록하고, step마다 `lag_versions`로 남긴다.

## LoRA

- `--lora expert` (기본, 사용자 10/28 "DAgger LoRA는 expert만으로 가자"): 모든 expert 층의 q k v o gate up down. VLM은 순전파만 한다(역전파 없음).
- `--lora llm+expert` (보류, 코드는 동작): 위와 같은 선형층을 모든 LLM 층에도 단다.
- 카메라·LiDAR ViT, 임베딩, LM head, action proj는 얼린다.
- 파일 형식은 `rl` 그대로다: `lora/vNNNNN.safetensors` + `latest.json`, 키 `<모듈>.lora_A/B`, metadata `scale`, `version`.
  - 모듈 이름은 납품 state dict, `quant_manifest.json` 이름과 1:1이다.
  - `rl.policy.merge`는 LLM 층에도 그대로 동작한다(테스트).
- 크기(실측, rank 16): expert 52 MB. llm+expert는 아래 표.
- 액터 엔진 `--lora-watch`는 지금 expert만 병합한다. LLM 쪽 확장은 다른 세션 일이다.

## 공개, 메트릭, 재개

- **첫 갱신**: DAgger 행이 `--min_rows` 이상 쌓인 뒤 시작한다.
- **공개 전 검사** (`--publish_min`마다): 고정 행으로 손실을 잰다. 고정 행은 R2 ft val `--val_n`개(eval 모드, 샘플 키로 정한 노이즈)와 이번 실행의 첫 DAgger 행 `--val_dagger`개(시드 고정)다.
  - 다음 중 하나면 공개하지 않는다: 값이 유한하지 않음, LoRA 가중치가 유한하지 않음, val fm·traj_ce가 버전 0의 `--val_max_ratio`(1.5)배 초과.
  - 기록용 세트(`dagger.val_sets`, 차단하지 않음): `signal_approach`와 `signal_hold`(R1 val의 같은 규칙 행 32개씩), `lane`(곡선 32 + R2 D 32 = ft_select의 curve 세트). `val_signal_approach/*`, `val_signal_hold/*`, `val_lane/*`로 남는다.
  - 공개하지 않으면 `alert` 이벤트를 남기고 `--alert_cmd`(예: Discord DM)를 실행한다.
  - `--max_rejects`(3)번 연달아 실패하면 저장하고 종료 코드 3으로 끝낸다.
- **이력 의존도** (사용자 10/28, 기록만 하고 공개는 막지 않음): 시작 납품본(v0)에서 한 번, 그 뒤 공개할 때마다 LoRA 파일을 내보낸 뒤 잰다. 정의는 `scripts/ft_measure.py history`와 같다.
  - 행마다 같은 초기 노이즈(seed 1000 k)에서 결정적 Euler 10 step을 돈다. 적분은 늘 그 행의 실제 이력으로 한다.
  - base: 실제 이력. zero: 이력 토큰을 정지(16점 원점)로 바꾼다. swap: 같은 세트의 행 (k + n/2) % n 이력을 쓴다.
  - 조건마다 base 대비 `move`(6.4 s 궤적 사이 평균 거리), `dlat_2s`·`dlat_3s`(GT 경로 기준 횡 차이)를 잰다.
  - 세트(`dagger.history_sets`): general(val 32), curve(곡선 16 + R2 D 16), highway(고주로 32, `assets/dagger/highway.parquet` = nav_text "No speed limit")
  - 기록: `history.jsonl`(버전별 세트 평균·p90, 걸린 시간), `history_rows.jsonl`(행별). 다른 메트릭과 함께 B2에 올라간다.
  - 참고 기준값(`reports/r2ft_regression_analysis.md`): s2 zero_move 4.43 m, R2 ft step 1000 0.84 m
  - 옛 납품본(`wowmayo_final`) v0 실측(5080, 압축, 세트당 32행):

| 세트 | zero_move | zero_dlat_2s / 3s | swap_move | swap_dlat_2s / 3s |
|---|---|---|---|---|
| general | 3.60 m | 0.81 / 1.45 | 7.59 m | 0.67 / 1.33 |
| curve | 10.14 m | 2.26 / 4.70 | 6.13 m | 0.65 / 1.43 |
| highway | 9.40 m | 1.93 / 3.82 | 5.93 m | 0.82 / 1.43 |

  - 걸린 시간: 5080 압축 경로에서 96행에 1,151 s(행당 12 s)다. 8×H100(압축 없음)은 GPU당 12행이라 약 15–20 s로 추정한다(미측정). 실제 값은 dry run의 `history.jsonl` `seconds`로 본다. 10분 공개 주기에 비해 길면 `dagger.history_sets`의 n을 줄인다.
- **유휴 정지** (`--stale_stop_min` 30): 새 행이 30분 동안 없고 밀린 행도 없으면 멈춘다(유휴 GPU 과금 방지). 상자는 idle watchdog이 세운다.
- **메트릭** (`<out>/metrics/`):

| 파일 | 내용 |
|---|---|
| `steps.jsonl` | step마다: 소스·손실별 값(`replay/<소스>/<손실>`, `dagger_planner|route/fm|traj_ce`), grad_norm, lr, step_s, 샘플/s, DAgger 행/s, data_wait, 새 행·재사용 행, 소비 행, 밀린 행, lag, peak_mem |
| `events.jsonl` | start, waiting_rows, first_update, publish, alert, resume, bad_episode |
| `val.jsonl` | 공개 후보마다 val 값과 실패 사유 |
| `history.jsonl`, `history_rows.jsonl` | 공개 버전마다 이력 의존도(세트별, 행별) |
| `spool.jsonl` | 에피소드마다 판단 수, 4 Hz 행, 라벨 없음, valid 없음, 소스별 행, 태그별 행, 신호 필드가 있는 행 |
| `summary.json` | 종료 때: 중앙값 step_s, 샘플/s, 최대 메모리, 균형 액터 수(`actors_at_balance`, 기준은 `actors_basis`) |

  - `run.sh`가 `tools/vast/metrics_sync.sh`로 B2 `runs/dagger/<RUN>/metrics/`에 2분마다, 그리고 끝날 때 올린다.
- **재개**: 같은 명령으로 이어 간다. `state.pt`에 LoRA, AdamW, step, 버전, replay 위치, 읽은 에피소드 순서, FIFO 위치, val 기준값이 있다.
- **DDP**: torchrun rank마다 모델 전체를 올린다. step마다 LoRA 기울기를 all-reduce(평균)한다. 저장할 때마다 rank별 LoRA 체크섬이 같은지 확인한다.
- **멈추기**: `touch <out>/STOP`(저장하고 끝남).

## 사용법 (cwd `train/`)

```bash
# 풀 다시 만들기 (Mac: R1 인덱스, _ego, scorer 지도)
python -m dagger.pools r1 --index ~/coc_runs/index_v3/index.parquet --ego ~/.cache/step0_ego \
  --start-relabel <step0>/_start_relabel/start_relabel_index.parquet --out assets/dagger
# R2 A·B 동적 풀 (B2 step0_r2의 net 인덱스·_meta, D: 또는 B2 .bak의 옛 인덱스)
python -m dagger.pools r2 --index <net index> --ego <step0_r2>/_ego --meta <step0_r2>/_meta/r2_episodes.parquet \
  --pools assets/ft --old-index <옛 step 인덱스> --r2logs <수집 폴더의 *.r2.log> --out assets/dagger   # Mac: 매니페스트·scorer 지도
# 단위 테스트, CPU 한 바퀴
TINY_DELIVER=<tiny deliver> E2E_SERVER=<eval 레포 src/e2e_driver/server> PYTHONPATH=$PWD:$PWD/tests python -m pytest tests/test_dagger.py
bash dagger/tiny_test.sh <tiny deliver> <work dir> <e2e server> [python]
```

**8×H100 상자**: R2 ft와 같은 준비를 쓴다(`tools/vast/r2ft_stage.sh`: R1·R2·venv·HF). 코드 tar에 다음 두 가지를 더한다. 이 레포의 `train/`·`tools/vast/`, 그리고 eval 레포의 `src/e2e_driver/server/assemble.py`와 `src/e2e_driver/src/e2e_driver/`이다. 둘째는 `/workspace/e2e/server`, `/workspace/e2e/src/e2e_driver`에 둔다.

- **dry run** (R1을 받는 동안): replay 몫 50 %를 R2(0.25/0.125/0.125)로만 채우고, R1 val 세트는 비어 건너뛴다. 가짜 spool 생성 → 로드 → 20 step → `metrics/summary.json`(샘플/s, 최대 메모리, 균형 액터 수) → 검사 공개 1회.

```bash
cd /workspace/train && BASE_B2=b2://morai2026-alpamayo/<납품 경로>/deliver/final bash dagger/dryrun.sh   # expert (기본)
```

- **본 실행** (출발값은 사용자 승인 뒤):

```bash
RUN=<이름> BASE=/workspace/ckpt/deliver/final LR=.. RANK=.. ALPHA=.. BATCH=.. PUBLISH_MIN=.. MIN_ROWS=.. \
  setsid nohup bash dagger/run.sh > /workspace/dagger_run.log 2>&1 < /dev/null &
```

- **Brev 예열** (`hyperstack_H100x8_NVLINK`, $24.96/h, 기동 약 8분 30초, 정지 기능 없음):

```bash
NAME=dagger-learner bash train/dagger/brev_up.sh          # Mac; 뒤로 ssh dagger-learner tail -f /workspace/stage.log
```

  - `brev_up.sh`는 상자를 만들고 코드(train/, tools/vast, tools/scorer, eval 레포 엔진 조립 코드)를 올린다. b2 계정 파일은 winpc에서 상자의 root 설정으로 파이프로 바로 보낸다. 키는 명령줄, Mac 파일, 로그 어디에도 남지 않는다.
  - `brev_stage.sh`(상자, root)는 `/workspace`를 가장 큰 디스크에 둔다. python3.12가 없으면 uv로 받아 venv 경로(`/usr/bin/python3.12`)에 연결한다.
  - 인수 확인: CPU(파이썬 1e7 루프, MHz) → `learner_meta/cpu.json`, B2 속도(venv tar) → `b2_speed.json`, `gpu_check` 8장.
  - 받기(병렬): venv, R2, R2 납품본(`runs/r2ft/deliver/final`, 있으면 `runs/r2stop/deliver/final`도), HF, R1(약 190 GB)
  - 필요한 부분이 다 오고 `gpu_check`가 끝나면, R1을 받는 동안 `dryrun.sh expert`(8장)를 돌린다.
  - 메트릭은 2분마다 B2 `runs/dagger/learner/<RUN>/`로 올린다. 학습은 시작하지 않는다.
  - Brev 상자는 안에서 스스로 멈출 수 없다(vast·RunPod용 `idle_watchdog.sh`는 동작하지 않음). 감시와 삭제는 상자를 가진 Mac 세션이 ssh로 한다.
- 액터는 B2 `runs/dagger/<RUN>/spool/<actor>/epK/`에 올린다. 러너는 각 버전을 B2 `runs/dagger/<RUN>/lora/`에 파일 먼저, `latest.json` 나중 순서로 올린다.
- 베이스는 `BASE`로 고른다. R2 납품본이 폐루프에서 떨어지면 옛 납품본 B2 `runs/stageC/deliver/final`을 쓴다.

## 제안 출발값 (사용자 승인 필요, 코드에 기본값 없음)

| 값 | 제안 | 이유 |
|---|---|---|
| `--lr` | 5e-5, warmup 20 step 뒤 고정 | LoRA 지도 학습의 보통 범위(1e-4)의 절반. 몇 분마다 액터에 나가므로 보수적으로 잡았다 |
| `--rank` / `--alpha` | 16 / 32 (scale 2) | `rl` learner 기본값과 같다. expert 52 MB |
| `--batch` | 32 (DAgger 16 + replay 16), `--micro 1` | R2 ft 전역 배치(8 GPU × 1 × 누적 4)와 같다. 몫을 바꾸면 DAgger·replay 수가 각각 8의 배수가 되게 잡는다(예: 0.4면 batch 40) |
| 몫·가중치 | 위 표 | 확정: 감속·접근 10 %, 정지 유지 5 %, 행 가중치 ×2. 미확정: DAgger 50 %, R1 일반 25 %, R2 5/2.5/2.5 % |
| `--publish_min` | 10 | 검사 1회는 8장에서 약 10 s로 추정한다(5080 1장에서 16행에 31 s). 엔진 병합은 0.8–0.95 s다 |
| `--min_rows` | 2,500 | 액터 1대가 한 바퀴 달리면 약 2,500행이다(4 Hz × RTF 0.84 × 12.6분) |
| `--lora` | `expert` (확정, 기본값) | 사용자 10/28. `llm+expert`는 보류 |

## 검증 (10/28)

- **풀** (10/28, R1 학습 행 1,818,540):
  - 감속·접근 55,478행(2 Hz 11,114): brake 49,467, 정지선 60 m 안에서 움직이는 행(cruise 5,792 + accel 219)
  - 정지 유지 379,051행(2 Hz 75,840)
  - 감속·접근은 2 Hz 행이 적다. 10 % 몫이면 schedule 한 바퀴에 같은 행이 약 5번 나온다(R1 소스는 MixtureDataset이 2 Hz 행에서만 뽑는다).
  - 미래 녹색 제외: 학습 2 Hz 364,140행 중 53,254행(14.6 %). to_green 17,813, unknown 35,441.
  - val+val_asset: 감속·접근 2 Hz 2,747행, 정지 유지 18,780행.
  - 자세한 수는 `assets/dagger/pools.json`에 있다.
- **단위 테스트** (`tests/test_dagger.py`, Mac): 12개 모두 통과했다(이력 측정의 결정성·짝 행·요약 포함). 기존 `test_configs.py`·`test_finetune.py`도 통과했다(풀 분리 뒤 `test_dagger` + `test_configs` 20개 다시 통과). 처음 7개는 winpc CPU에서도 통과했다.
  - 태그·가중치 규칙, 미래 녹색 규칙(합성 배열), 행 가중치가 손실에 곱해짐, `dagger.yaml` 구성
  - 4 Hz 선택, 라벨 검사, 가짜 에피소드 행 수, sync 중 에피소드 건너뛰기
  - DAgger 손실 = R2 GT 행 손실(같은 noise)
  - valid 마스크(invalid 점의 값을 바꿔도 손실이 같음)
  - 체크포인팅 전후 기울기 일치(llm+expert, replay 경로 포함)
  - LLM LoRA 이름 = 납품 이름, merge
- **CPU 한 바퀴** (`tiny_test.sh`, winpc CPU, 양자화기 끔): PASS.
  - 1프로세스: 대기 → 첫 갱신 → 공개 v1·v2 → 재개(replay 위치 연속) → v3
  - gloo 2프로세스 llm+expert + 체크포인팅: 저장 때마다 LoRA 일치, v1 공개, merge 56층
- **실제 설정** (winpc 5080, `--compress`, 실제 R1·R2(D:), 가짜 DAgger 행, batch 4, 행 가중치 ×2로 시험; 풀을 둘로 나누기 전): 끝까지 돌았다.
  - 혼합: replay 안에서 r1_general 0.5, signal_stop 0.3, new_a 0.1, new_b·new_d 0.05. 배치 기준 몫과 같다.
  - R1 일반 replay 제외 목록이 인덱스 382,575행을 뺐다(10 Hz 전 split, 출발 재라벨 + 미래 녹색).
  - 신호 정지 학습 2 Hz 행은 84,019개다(4개 에피소드는 step-0 파일이 없어 MoraiDataset이 뺐다).
  - 버전 0의 기록용 val: `val_signal_stop` fm 0.302 / traj_ce 0.227, `val_lane` fm 1.539 / traj_ce 2.544 (옛 납품본, 8행씩).
  - step 기록에 `tag_signal_rows`, `tag_lane_rows`, `dagger_signal/*`, `replay/<소스>/*`가 들어갔고, v1을 공개했다.
- **`rl.tiny_test`** (덧붙인 `LoRA` 인자, winpc CPU, 12 업데이트 × 4 에피소드): PASS. 기존 RL 경로는 그대로다.
- **`dryrun.sh` 자체** (winpc 5080, `NG=1 BATCH=4 STEPS=4 DRY_EXTRA=--compress`): 끝까지 돌았다.
  - R2 전용 혼합(new_a 0.5, new_b 0.25, new_d 0.25) → R2 val 32행 → 4 step → 검사 → v1 공개 → summary
  - 몫 오버라이드 방식으로 바꾼 뒤에도 다시 돌렸다(10/28 07:47): 배치 기준 DAgger 0.5 + R2 0.25/0.125/0.125. 기록용 val은 signal_stop 0행(건너뜀), lane 3행(R2 D)이었다.
  - WSL에서는 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`가 필요하다. 켜면 step 2의 NVFP4 역양자화에서 드라이버 리셋이 두 번 났다. 리눅스 학습 상자는 R2 ft처럼 켜 둔다.
- **5080 실측** (실제 납품물 `wowmayo_final`, 실제 R1·R2 replay(D:), 가짜 DAgger 행, `--batch 4`):
  - 그대로 올리면 가중치만 13.3 GiB라 여유가 0.3 GiB다. 학습이 들어가지 않는다.
  - 그래서 `--compress`(ModelOpt `mtq.compress`)를 썼다. 5.4 GiB가 되고, NVFP4 층 출력은 정확히 같다. FP8 층은 실제 FP8 GEMM을 타서 최대 약 2 % 다르다.
  - 이 경로는 역양자화가 GPU 시간의 약 70 %를 차지한다. 그래서 아래 속도는 H100 처리량을 예측하지 못한다. H100은 `--compress` 없이 돌리고, 처리량은 dry run으로 잰다.

| 설정 | step_s (4샘플, 중앙값) | 샘플/s | 학습 중 최대 메모리 | 공개 검사 (16행) | LoRA (rank 16) |
|---|---|---|---|---|---|
| expert, `--compress` | 8.44 | 0.47 | 11.54 GB | 31 s | 52 MB |
| llm+expert, `--grad_ckpt --compress` | 32.5 | 0.12 | 12.45 GB | 51 s | 113 MB |

- 첫 step의 14.05 GB는 적재·압축 중 순간값이다(압축 전 가중치 13.3 GiB).
- 프로파일(DAgger 행 1개): ViT 0.25 s, VLM prefill 0.61 s, DAgger step(fwd+bwd) 1.28 s. 입력 양자화를 꺼도 1.17 s라, 시간은 대부분 압축 가중치의 역양자화(copy·index·mul)에 든다.

## 위험

- 5080 숫자는 압축 경로의 숫자다. H100 메모리 추정(미측정)은 다음과 같다.
  - expert: 5080 값 − 압축 가중치 5.4 + BF16 가중치 13.3 ≈ 20 GB
  - llm+expert(체크포인팅 없이): LLM 활성값과 저장되는 가짜 양자화 가중치가 더해져 약 40 GB
  - 둘 다 80 GB 안이다. 실제 값은 dry run의 `peak_mem_gb`로 본다.
- 러너는 8×H100이고, 액터를 늘려 전체 DAgger 시간을 줄인다(사용자 10/28). 균형 액터 수는 DAgger 행/s ÷ 3.36이다. 8장 + expert 설정 dry run(`dryrun.sh` 기본값)의 `summary.json` `actors_at_balance`로 정하고, 상한은 두지 않는다. 러너가 더 빠르면 `uses_*`로 재사용 정도를 본다.
- 라벨 파일 형식은 라벨러 세션과 아직 맞춰 보지 않았다. 가짜 라벨로만 시험했다. `signal`·`stop_d`가 오기 전에는 신호 태그가 0이다.
- H100 경로(`flash_attention_2`, 압축 없는 가짜 양자화, NCCL 8장)는 아직 돌려 보지 않았다. dry run 첫 몇 분에 드러난다. FA2에서 막히면 `ATTN=sdpa`로 바로 돌린다.
- B2 업로드(`--b2`)는 명령 문법(b2 5.0.2 `file upload --no-progress BUCKET LOCAL NAME`)만 확인했다. 실제로 올려 보지는 않았다.
