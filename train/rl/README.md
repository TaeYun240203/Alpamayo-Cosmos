# 고주로 PPO learner (RL_PLAN 3.2–3.5, 4.2)

와우마요 납품물(QAD `final/` + `modelopt_state.pth`)을 actor 엔진과 같은 NVFP4/FP8 가짜 양자화 상태로 얼려 올린다. expert 선형층에만 BF16 LoRA를 달아 Decoupled PPO로 학습한다.
- 카메라·LiDAR ViT와 LLM은 순전파만 한다.
- critic은 시뮬레이터 GT 벡터를 입력으로 받는 특권 critic이다(사용자 10/27 B안).
- `morai_vla`(StudentVLA, processor, collate)를 import해서 쓴다. 기존 파일은 고치지 않았다.

| 파일 | 내용 |
|---|---|
| `policy.py` | 베이스 로딩, LoRA, 체인 log-prob 재계산, 기준 actor 샘플러 `Policy.sample`, LoRA 내보내기, `merge` |
| | LoRA는 forward hook으로 단다. 그래서 모듈 이름·양자화기·state dict가 납품물 그대로다 |
| `episodes.py` | learner 입력 형식(파일 머리말이 명세), 보상, 판단 단위 Dataset |
| `learner.py` | 특권 critic, 웜스타트, PPO 업데이트, 늦음 게이트, LoRA 버전 공개, 재개 |
| `tiny_test.py` | 소형 모델 시험 (아래) |

## 입력 (actor → learner)

에피소드 폴더 하나 = `steps.jsonl` + `done.json`이다. steps.jsonl은 판단 하나가 한 줄이다.
- **엔진 덤프**: actor 엔진(`model_server --dump`, eval 브랜치 `rl/actor-engine`)의 `req_ID.bin`/`out_ID.bin` 파일 이름이 들어 있다.
  - learner는 req를 엔진의 `assemble.build_sample`로 모델 입력으로 만든다. 그래서 엔진과 같은 프롬프트를 본다.
  - out에서 expert, lora_version, chain, sigma, logp를 읽는다.
- **rl-env 필드**: `rl_reward.py`의 같은 판단 행이다. r_time, r_penalty, r_terminal, phi, phi_next, events, critic_obs.
- 폴더 조립은 rl-env 세션이 맡는다(형식은 그 세션에 보냈다).
- PPO 손실에는 고주로 expert 벌(`out.expert == --train_expert`, 기본 1)의 판단만 쓴다.
  - CP10 전과 CP13 뒤의 기본 expert 판단은 환경으로 본다. 보상과 가치는 GAE에 그대로 들어간다.
- 데이터는 버리지 않는다. done.json 없이 `actor_timeout` 동안 멈춘 폴더(actor가 죽은 경우)도 잘린 에피소드로 학습한다.

## 정책과 손실

- RL 주행의 행동은 노이즈 Euler 체인이다: x₀ ~ N(0, I), x_{k+1} = x_k + dt_k·v(x_k, t_k) + σ_k·ε. 행동은 x_K이고, 정규화 (a, κ) 공간 64×2다.
- log π = Σ_k log N(x_{k+1}; x_k + dt_k·v, σ_k²)이다.
  - learner는 판단마다 VLM prefill을 한 번 하고, KV를 K번 복제해 K스텝을 expert 한 번에 다시 계산한다.
- 손실 (`learner.ppo_loss`): 가중치 w = min(π_prox/μ, w_max)를 곱하고, 비율 r = π_θ/π_prox를 [1−ε, 1+ε]로 자른다.
  - μ는 actor 기록값이다(NVFP4, 1–2 버전 전). π_prox는 업데이트 시작 시점 learner의 값이다.
  - 정밀도 차이와 늦음은 w가 흡수한다. 로그의 `behav_gap_*`과 `w_truncated`가 그 크기다.
- SFT 기준 KL은 `--kl_coef > 0`일 때만 켜진다. LoRA를 끈 expert와의 가우시안 KL을 닫힌 꼴로 계산한다.
- **늦음 게이트**: actor는 에피소드를 시작할 때마다 `latest.json` 버전을 쓰고, `<spool>/<actor>/inflight.json`을 쓴다.
  - learner는 업데이트 n → n+1 전에, 살아 있는 모든 actor가 버전 n+1−max_lag 이상을 쓸 때까지 기다린다(기본 2).
- **보상**: r_time + r_penalty + r_terminal + γ·phi_next − phi다. rl-env의 r_shape는 γ=1인 경우이고, γ 0.995에 맞게 다시 계산한다.
  - GAE λ는 0.95다. terminated면 0, truncated(시간 제한, actor 죽음)면 마지막 V로 bootstrap한다.
- **critic (B안, 사용자 10/27)**: rl-env가 판단마다 기록하는 GT 약 30개 값(`critic_obs`)을 입력으로 받는다.
  - 구조: running mean/std 정규화(±10에서 자름) → MLP 2×256(tanh) → V. 이미지·VLM은 쓰지 않고, 배포하지 않는다.
  - 학습: 옵티마이저는 따로 둔다(`--critic_lr 1e-3`, 정책 1e-5). 업데이트마다 GAE 리턴에 MSE를 `--critic_epochs 4`번 맞춘다.
  - 웜스타트(`--warmstart <기준 주행 spool>`): 납품 모델 주행의 λ=1 리턴(Monte-Carlo, 잘린 에피소드는 현재 V로 bootstrap)에 `--warm_epochs`만큼 맞춘다.
    - 이름순으로 10개마다 1개를 떼어 두고 explained variance를 `warmstart.json`에 쓴다.
  - 업데이트마다 `value_ev`도 기록한다. 이번 학습 전 가치로 새 에피소드(학습에 안 쓴 데이터)에서 잰 값이다.

## 사용법 (cwd `train/`)

```bash
python -m rl.learner --base <deliver>/final --spool <spool> --out <out> --e2e_server <eval 레포 src/e2e_driver/server> --warmstart <기준 주행 spool> --device cuda --micro 2 --minibatch 64
```

- 처음 시작하면 웜스타트를 하고, `<out>/lora/v00000.safetensors`(영 LoRA = 납품 expert)와 `latest.json`을 공개한다.
- 업데이트마다 `vNNNNN.safetensors`, `state.pt`(재개용), `log.jsonl`을 쓴다. 같은 명령을 다시 실행하면 이어서 돈다.
- 기본값(출발값, RL_PLAN 9절 미정): `--lr 1e-5 --rank 16 --alpha 32 --clip 0.2 --w_max 2 --epochs 1 --kl_coef 0 --gamma 0.995 --lam 0.95 --min_decisions 256 --max_lag 2`.
- LoRA 파일은 BF16 safetensors `{<module>.lora_A (r, in), <module>.lora_B (out, r)}`이고, metadata에 `scale = alpha/r`와 `version`이 있다.
  - 실제 크기에서 rank 16이면 약 66 MB다(추정). actor 엔진은 `--lora-watch`로 받아 합치고 다시 양자화한다.
- 납품 형식 폴더 만들기:

```bash
python -c "from rl.policy import merge; merge('<deliver>', '<out>/lora/v00012.safetensors', '<new deliver>')"
```

  - expert 가중치만 W + scale·BA로 바뀐다. amax와 `quant_manifest.json`은 납품값 그대로다. 납품 amax는 max|W|가 아니라 QAD 보정값이라, 같은 격자로 다시 양자화한다.

## 소형 시험 (`tiny_test.py`)

```bash
HF_HUB_OFFLINE=1 python -m rl.tiny_test <tiny deliver dir> <work dir> --e2e_server <eval 레포 src/e2e_driver/server> [--device cuda] [--updates 12 --episodes 4]
```

- 베이스: `student_dryrun.sh`(TINY=1)의 소형 납품물이다. 형식은 실제 납품물과 같다(NVFP4 21층 + FP8 51층, manifest).
  - 위치: winpc `~/morai-student/runs/dry/deliver`, Mac 사본 `~/claude-bench/rl-tiny/deliver`.
- 입력: 엔진 덤프 형식(wire req/out)의 가짜 주행이다. 무작위 카메라 JPEG과 LiDAR 스캔을 쓰고, 판단 0은 기본 expert, 나머지는 고주로 벌이다.
- 가짜 보상: 행동의 평균 정규화 가속도. `critic_obs[0]` = 남은 판단 수이고, 나머지 29개는 잡음이다.
- actor는 learner보다 1 버전 늦은 BF16 LoRA 파일로 주행한다.
- 검사 항목:
  1. 버전 0에서 learner 일괄 재계산 log-prob = actor 순차 log-prob
  2. 결정적 Euler 가속도와 6.4 s 진행 거리가 오른다(업데이트 8회 이상일 때)
  3. `merge` 결과 = 베이스 + 내보낸 LoRA. 기록된 체인의 스텝별 log-prob로 비교한다.
     - 끝 궤적으로 비교하지 않는 이유: 소형 모델은 혼돈적이라, 가중치를 1e-7만큼 상대 변화시켜도 행동이 약 0.02 움직인다.
  4. 루프: 웜스타트(떼어 둔 에피소드 EV), 공개, 게이트 대기, `state.pt` 재개, done.json 없는 에피소드 학습
- CPU에서는 양자화기를 끈다. ModelOpt NVFP4 가짜 양자화가 CUDA 전용이기 때문이다.

**결과: Mac CPU, 양자화기 끔, 10/27** (`logs/rl/tiny_cpu_summary.json`)
- 조건: 업데이트 12회, 업데이트마다 에피소드 4개 × 판단 8개, 정책 판단 28개. lr 2e-3, epochs 2, σ 0.3.

| 항목 | 값 |
|---|---|
| 1. 버전 0 log-prob 차이 | 0.000 (behav_gap 평균 0, 표준편차 0). 별도 스모크에서 순차 vs 일괄 4e-6 |
| 2. 결정적 가속도(정규화) | 0.085 → 0.420 |
| 2. 6.4 s 진행 거리 | 63.18 → 66.11 m |
| 2. 노이즈 주행 평균 가속도 | 0.06 → 0.31 |
| 3. `merge` vs 베이스 + LoRA 파일 | 판단당 log-prob 차이 최대 0.004 (log-prob 약 −284) |
| 4. 웜스타트 | 떼어 둔 에피소드 EV 0.95 |
| 4. 업데이트별 value_ev | −0.19–0.83 (가짜 보상의 잡음이 큼) |
| 4. 루프 | 공개 v0→v2, 재개, 게이트 1.04 s 대기, done.json 없는 에피소드 포함 |

- clip 비율은 0.3–0.6이었다(lr 2e-3, log-prob 1280차원 합). actor가 1 버전 늦을 때 behav_gap 표준편차는 0.4–0.9였다.
- 속도는 CPU에서 판단당 약 3 s다(이미지 조립 포함).
- 실제 엔진 요청으로도 확인했다(`assets/warmup/req_000600·601.bin`: 카메라 4대 × JPEG 4장, 스캔 4개, 실제 경로 문장).
  - `Decisions` → 프롬프트 1,728토큰 → learner 재계산 log-prob 차이 0.0.
- CUDA에서 양자화기를 켠 시험은 아직 하지 않았다. 사용자가 winpc 사용 중지를 지시해서 확인을 기다린다. 준비된 명령(winpc):

```bash
cd ~/morai-rl/train && HF_HUB_OFFLINE=1 PYTHONPATH=$PWD:$HOME/morai-student/pylib ~/claude-bench/venv-teacher/bin/python -m rl.tiny_test ~/morai-student/runs/dry/deliver ~/morai-rl/work_cuda --e2e_server ~/morai-rl/e2e/server --device cuda --updates 12 --episodes 4
```

## 위험·측정할 것

- **log-prob 차원**: log-prob가 10스텝 × 128차원의 합이다. 그래서 작은 가중치 변화에도 비율이 크게 움직인다(소형 시험 lr 2e-3에서 clip 비율 0.4–0.6).
  - 실제 크기에서 `clip_frac`과 `approx_kl`을 보고 lr을 정한다.
  - 계속 크면 DPPO처럼 스텝마다 비율을 자르는 방식으로 바꾼다. 이건 설계 변경이라 사용자 확인이 필요하다.
- **actor–learner 차이**: NVFP4 엔진 커널과 ModelOpt 가짜 양자화의 차이, 합친 LoRA를 다시 양자화할 때의 반올림이 있다.
  - pod에서 버전 0 `behav_gap_std`를 잰다. 1을 크게 넘으면 w가 잘리기만 해서 학습이 치우친다.
- **처리량**: 업데이트마다 VLM prefill을 (1 + epochs)번 다시 한다. 얼린 VLM의 KV는 판단 하나에 수백 MB라 캐시하지 않는다.
  - KV를 K번 복제하므로 `--micro`로 메모리를 맞춘다.
  - 5090 25–30 샘플/s는 계획서 추정이고 아직 재지 않았다.
- learner 전 과정은 sdpa로 돈다. 복제 행에 4D 마스크가 들어가는데 FA2가 거부해서다.

## 10/28 추가: 여러 GPU, B2, 실행 스크립트 ([PPO_PLAN.md](../../PPO_PLAN.md) 10절)

- **여러 GPU:** `torchrun --nproc_per_node N -m rl.learner ...`로 띄운다.
  - 랭크마다 정책 판단을 나눠 맡는다. 나누는 방식은 버전에서 정해지므로 모든 랭크가 같다.
  - LoRA 기울기는 평균한다. 최적화 스텝 수는 모든 랭크가 같다.
  - 에피소드를 기다리고, critic을 맞추고, 공개하는 일은 rank 0만 한다. critic과 첫 LoRA는 rank 0의 값을 나눠 준다.
  - 업데이트마다 모든 랭크의 LoRA 체크섬이 같은지 확인한다.
- **`--b2 b2://bucket/prefix`:** 버전마다 `vNNNNN.safetensors`를 먼저 올리고 `latest.json`을 나중에 올린다(`<prefix>/lora/`).
- **`--kl_target`:** 적응형 KL. 목표는 판단당 nats이고, 시작 계수는 0.1/목표다. 사용자 10/28: 스텝마다 0.1–0.2σ 이동 = 6.4–25.6.
- **`--stale_stop_min`:** 기본 30분. 새 에피소드가 이만큼 없으면 저장하고 멈춘다.
- **로그 추가:** `ranks`, `peak_mem_gb`, `s_per_policy_decision`, `kl_coef_next`. behav_gap은 모든 랭크를 합친 값이다.
- **스크립트:**

| 파일 | 내용 |
|---|---|
| `fake.py` | 실제 크기의 가짜 PPO spool(`dagger.fake` 재사용, critic_obs 50값) |
| `dryrun.sh` | 다운로드 중 속도 측정. 업데이트 1번 → `summary.json`의 `actors_at_balance` |
| `run.sh` | 본 실행. B2 spool을 staging으로 받아 완료된 에피소드만 hard link로 옮긴다. 메트릭을 올리고 learner를 돌린다 |
| `ddp_test.py`·`ddp_test.sh` | CPU gloo 2 프로세스 시험: 업데이트 → 재개 → 업데이트, 랭크 간 LoRA 일치 |

- **시험 (Mac CPU, 10/28):** `ddp_test.sh` PASS. 첫 시험에서 랭크마다 LoRA A 초기값이 달라 가중치가 갈라지는 것을 잡아 고쳤다. `tiny_test` check 4(적응형 KL과 재개)도 PASS.
