# 교사 T\* 학습 코드 (MODEL_PLAN.md 2단계, 3단계 Stage 1-T·1-C·2)

Alpamayo-1.5-10B에 LiDAR ViT를 붙여 MORAI 데이터로 미세조정하는 코드다. NVlabs/alpamayo-recipes `recipes/alpamayo1_5_sft`(커밋 `2e907da`)와 그 recipe가 고정한 `alpamayo_r1`(NVlabs/alpamayo `f3198c2`)를 pip 패키지로 설치해 쓰고, upstream 파일은 하나도 고치지 않았다. `data.py`와 `processor.py`는 학생도 그대로 쓴다. 학생(Qwen3-VL-4B) 코드·검증·메모리 계획·런북은 아래 "학생" 절에 있다.

## 구성

| 파일 | 내용 |
|---|---|
| `morai_vla/build_index.py` | (Mac) ego 표 + 라벨 parquet + manifest → `index.parquet`, `splits.json`. 행 = τ 격자 0.1 s 한 점. CoC는 넣지 않는다(설정한 parquet을 데이터셋이 읽는다) |
| `morai_vla/lidar_bev.py` | (vast 세션 작성) VLP16 디코드, GT 속도·yaw rate 적분 자세, 스캔 내 움직임 보정, BEV 래스터, 2스캔 묶음(`pack_bev`). step 0 worker의 `lidar` 모드가 이것으로 에피소드마다 `lidar.bin` + `lidar_index.npz`(스캔별 zlib 블롭)를 쓴다 |
| `morai_vla/data.py` | `MoraiDataset`: τ 프레임·스캔 선택(지터), BEV 패킹·워프, 이력·미래 궤적, nav_text, LiDAR dropout, 셀 하위 샘플링, recipe의 CoC 끔/켬 모드 |
| `morai_vla/processor.py` | 1.5 대화 형식의 MORAI판(카메라별 픽셀 수, 시각 텍스트, BEV 2장, Stage 1-C의 CoC). 토크나이저 설정과 collate는 recipe 것을 그대로 쓴다 |
| `morai_vla/model.py` | `TeacherVLA`: `AlpamayoR1` + LiDAR ViT(카메라 ViT·merger·DeepStack merger 3개의 사본). 단계별 학습 부위와 손실, 추론(`sample_trajectories`) |
| `morai_vla/train.py` | 학습 진입점. recipe `train_hf`와 같고 `resume`만 더했다 |
| `configs/` | `base`(recipe Stage 1 값), `step2`, `stage1t`, `stage1c`, `stage2`, `tiny_overfit`(로컬 검증), `deepspeed/zero2*.json` |
| `tests/` | 토큰 배치·개수, τ 선택, BEV 워프, 라벨 정규화 왕복, 실행 설정과 런북 오버라이드의 구성 검사, 복사 중인 에피소드 거르기, Stage 1 모니터의 이산 토큰 디코딩·클램프 집계, 학생(`test_student.py`, 아래 "학생" 절) (42개, 10/25 winpc) |
| `scripts/check_release.py` | 실제 10B 가중치(또는 `--ckpt` 학습본)로 MORAI 샘플 추론, ADE/FDE/minADE를 등속 기준과 셀별로 (16 GB 카드는 NF4) |
| `scripts/bench_io.py` | 데이터로더 처리량 |
| `scripts/stage_b.sh`, `scripts/stage_b_queue.sh` | 단계 B 런북 1–7과 후속 측정을 vast 인스턴스에서 무인 실행 |
| `scripts/smoke_stage1.sh`, `scripts/reload_check.py` | Stage 1-T·1-C 실제 크기 스모크(ZeRO-2 + CPU AdamW) → 저장 → strict 재로드와 부위별 가중치 변화 |
| `scripts/lidar_onoff_ce.py` | 같은 val 샘플의 teacher-forced 궤적 CE, LiDAR 있음/강제 제거 비교 |
| `morai_vla/stage1_eval.py`, `scripts/eval_stage1.py` | expert 없는 Stage 1 모니터: LiDAR 있음/없음 val CE, 이산 궤적 토큰을 greedy 생성·디코딩한 ADE/FDE (아래 "Stage 1 모니터링") |
| `scripts/tokenizer_clip.py` | 이산 궤적 토크나이저가 train 미래 궤적을 [-10, 10] 밖으로 잘라 내는 비율 |
| `scripts/time_fwd_bwd.py`, `scripts/time_soft_targets.py` | Stage 1 순전파+역전파 샘플당 시간, soft target 생성 시간(K 샘플 Euler, 고정 (x_t, t) 속도장) |
| `scripts/diag_native_prompt.py`, `scripts/diag_context_probe.py` | 로컬 진단(원본 1.5 입력 형식 비교, 이미지·이력 반사실 시험) |
| `setup_env.sh` | venv 설치 (PyPI, download.pytorch.org, NVlabs GitHub만) |

## 정한 것

**입력 시퀀스** (`processor.py` 머리말, `tests/test_layout.py`)
- `system | 카메라 이름, (시각, 이미지)×4 를 Front, Left, Right, Cam7 순 | "LiDAR BEV: ", (시각, BEV)×2 | 이력 48 | route | 지시문 || [CoC] <traj_future_start> 128 <traj_future_end>`
- 생성 모드 프리픽스는 **1,725토큰**이다(이미지 1,344 + BEV 100 + 텍스트·특수 토큰). 계획의 약 1,750과 맞는다.
- 카메라 이름: `Front camera`, `Front left camera`, `Front right camera`, `Front upward camera`. 좌우 카메라는 장착값상 70° 바깥, 10° 아래를 보므로 1.5의 front-left/right 이름을 그대로 쓴다. Cam7은 1.5에 맞는 이름이 없어(telephoto가 아님) 새 이름을 붙였다.
- 시각 텍스트는 프레임의 τ − t0를 소수 둘째 자리까지 쓴다: `<-0.31 seconds>`. 프레임은 오래된 것부터 놓는다. BEV에는 새 스캔의 시각을 붙인다.
- 카메라별 `min_pixels = max_pixels = W×H`로 처리해 크기가 그대로 유지된다(1.5 기본값 163,840이면 모든 MORAI 카메라가 확대된다). 토큰 수: Front 112, Left·Right 88, Cam7 48, BEV 50.

**시간축과 라벨**
- 프레임: τ 목표 t0, t0−Δ3, t0−Δ3−Δ2, … 에 가장 가까운 기록 프레임을 고르되, 바로 다음(더 새) 프레임보다 반드시 오래된 프레임으로 내린다. 시뮬레이터가 프레임을 떨군 구간에서 "가장 가까운 프레임"만 쓰면 **행의 9.9%에서 같은 프레임이 두 번** 들어갔다(실측, 35개 에피소드 5,143행). 실제 시각은 시각 텍스트가 알려 준다. 명목 목표(0.1 s 간격)에서 어느 프레임이든 0.08 s보다 멀면 그 행은 쓰지 않는다(1.8% 탈락). 학습 때 Δ ~ U(0.08, 0.12).
- 스캔 4개: 같은 규칙(8.4–8.9 Hz라 같은 스캔이 두 번 뽑힐 수 있다). BEV1 = (k1 ← k0), BEV2 = (k3 ← k2). 옛 스캔은 저장된 자세로 새 스캔 좌표계에 옮긴다. 자세는 GT 속도와 yaw rate를 τ로 적분한 값이다(GT x, y는 위 궤적과 같은 지터가 있고, 본선에는 GT 자세가 없다). 채널: 새 높이, 옛 높이(워프), 새 밀도×16(255에서 자름).
- 이력 16점(t0−1.5 … t0)과 미래 64점은 t0 자차 좌표계의 xyz·회전이다. (a, κ) 정규화는 모델 안의 `UnicycleAccelCurvatureActionSpace`(1.5 통계)가 한다.
- **위치는 GT 속도와 yaw를 τ로 적분해 만든다.** ego 표의 x, y는 Ego status 메시지를 도착 시각으로 τ 격자에 놓은 값이라, 이웃 행 사이 위치로 구한 속도가 ±30% 흔들린다(GT 속도 14.5–15 m/s일 때 11→14→17→11→20). 이걸 그대로 쓰면 정규화 가속도 표준편차가 9.9(PAI는 1)가 되어 FM 손실이 줄지 않았다. 적분으로 바꾼 뒤 정규화 가속도 표준편차 2.0, 곡률 0.74이고, 실제 MORAI 미래의 (a, κ) 왕복 오차는 평균 0.004 m, p95 0.010 m, 최대 FDE 0.064 m다(200샘플). 가속도가 PAI보다 큰 것은 특권 주행기의 GT 긴급제동 때문이다.
- GT 속도가 0.1 s 사이에 1.5 m/s(15 m/s²)보다 크게 변하는 지점이 이력+미래 창 안에 있으면 그 행을 뺀다. 어떤 제동으로도 나올 수 없는 값이고(충돌·속도 결측이 0으로 채워진 곳), 적분 궤적을 망가뜨린다. 12,810행(0.6%)이 빠졌다.
- `traj_ok`: 라벨의 traj_ok는 벽시계 [t, t+6.4 s]로 판정돼 있다. 시뮬레이터가 0.7–0.95배속이었으므로 τ 기준 [τ, τ+6.4]를 덮는 라벨 행이 모두 ok일 때만 쓴다. 이력 1.5 s와 미래 6.4 s가 없는 행도 뺀다. 인덱스에는 쓸 수 있는 행만 들어 있다(208.6만 행, 약 58 h).

**nav_text** (`build_index.py`): 150 m 안의 다음 기동을 1.5 형식으로 쓰고 속도 규칙을 붙인다. 예: `Turn left in 19m. Speed limit 60 km/h`.
- 기동: 교차로 `Turn left/right in Xm` · `Go straight through the intersection in Xm`(정지선 기준, 교차로 안에서는 0m), `Enter/Exit the roundabout in Xm`, `Take the ramp on the right in Xm`, `Merge left in Xm`, `Go through the toll gate in Xm`. 없으면 `Follow the road`.
- 속도: 예외 구간(s 1118.7–1741.7, 링크 A2256W000411–A2256W000153)은 `No speed limit`, 나머지는 `Speed limit 60 km/h`.

**분할** (`splits.json`, seed 0)
- 시나리오 시드(장면 변형 id, 없으면 에피소드) 단위 5%를 `val`로 둔다.
- 보류 에셋은 장애물 16종 중 2종(40100027, 40100029), 보행자 3종 중 1종(50100001)이다. 보류 에셋이 있는 장면에서는 그 액터 위치 s −120…+30 m 행만 `val_asset`으로 빼고 학습에 쓰지 않는다. 에피소드를 통째로 빼면 보행자 1종 때문에 전체 장면의 약 1/3이 학습에서 빠지기 때문이다.
- 결과: train 1,818,540행, val 115,333, val_asset 152,254. CP2·CP14 출발 `rs2041` 71개 에피소드는 다른 에피소드와 같은 규칙으로 들어간다(약 84k행이 train).
- CoC 키프레임(`use == True`) 13,386개 중 10,722개가 남았다. 나머지는 τ 기준 traj_ok가 거짓이거나 속도 글리치 창에 걸렸다.

**샘플링** (`base.yaml`): recipe의 CoC 모드 두 가지만 있고 섞지 않는다(`alpamayo1_sft` README "CoC reasoning enabled", AR1 §5.1–5.2).
- `coc: false`(step 2, Stage 1-T, Stage 2): 분할의 모든 사용 가능 행. `frame_stride 5`(2 Hz), 넘친 셀 유지 확률 `C0 0.4, C3 0.65, C5 0.55`(행 해시로 고정).
- `coc: true`(Stage 1-C): `data.coc_parquet`의 `use == True` 키프레임 중 인덱스에 있는 행만. t0는 키프레임의 τ0 그대로다(모든 τ0가 0.1 s 격자 위에 있음을 확인, 오차 1e-13). v1 기준 train 9,624개, val·val_asset 1,098개. 나머지 2,664개는 τ 기준 traj_ok가 거짓이거나 속도 글리치 창이라 궤적 감독을 할 수 없어 빠진다.
- CoC parquet 경로는 설정값이다(`data.coc_parquet`, 기본 `${data.root}/_coc/coc_dataset.parquet`). v1.1 재라벨은 경로만 바꾸면 되고 인덱스를 다시 만들 필요가 없다.

**LiDAR dropout**: `data.lidar_dropped(episode, i, p, seed)`가 샘플 키의 해시로 정한다. 학생·soft target 생성도 같은 (p, seed)로 이 함수를 부르면 같은 샘플에서 LiDAR가 빠진다. 빠진 샘플은 BEV 블록(이름·시각·이미지)이 통째로 없다. Stage 1-T·1-C·2는 0.15, step 2는 0이다(LiDAR만 학습하므로).

**단계** (`configs/`, 공식 recipe `alpamayo1_5_sft` @2e907da 기준, MODEL_PLAN 4.2–4.3)

| 실행 | recipe 단계 | 데이터 | 손실 | 학습 부위 | 설정 |
|---|---|---|---|---|---|
| step 2 | Stage 1 | `coc: false` | 이산 궤적 CE (+ 답의 끝 토큰 CE, recipe와 같음) | LiDAR ViT·merger·DeepStack (0.6B) | 2000 step으로 짧게 |
| Stage 1-T | Stage 1, CoC 끔 | `coc: false`, step 2에서 이어서 | 같음 | 카메라 ViT(merger 포함 0.1×), LiDAR 경로, LLM(임베딩 포함), LM head | |
| Stage 1-C | Stage 1, CoC 켬 | `coc: true`(키프레임만), 1-T에서 이어서 | + CoC LM (`cot` + `traj_future`) | 같음 | 200 step마다 평가 |
| Stage 2 | Stage 2 | `coc: false`(프리픽스에 CoC 없음) | flow matching, KV stop-gradient | expert, action_in/out_proj | lr 1e-4, warmup 100, 누적 1, DeepSpeed·checkpointing 없음 |

- 하이퍼파라미터는 recipe 값이다(`base.yaml`): Stage 1은 lr 1e-5, warmup 500, GPU당 배치 1, 누적 4, cosine(min 1e-6), 3 epoch, ZeRO-2, gradient checkpointing, `lr_multiplier {vlm.model.visual: 0.1}`. Stage 2는 recipe `sft_stage2_nav` 값.
- recipe와 다른 점은 하나다. Stage 2는 expert를 fp32로 두고 BF16 autocast로 계산한다(`master_fp32`). recipe는 DeepSpeed 없이 BF16 가중치를 바로 갱신하는데, 1e-4 갱신이 BF16 반올림에 묻힐 수 있어서다.
- Stage 1-T·1-C는 학습 파라미터가 약 9.5B라 AdamW 상태만 약 150 GB다. GPU 8장(ZeRO-2)이 필요하다. 1장이면 `zero2_offload.json`(CPU에서 AdamW, 느림, 미검증).
- Stage 2는 FM이 KV 캐시를 읽으므로 gradient checkpointing을 켜면 안 된다(HF가 캐시를 버린다). 설정 검사 테스트가 이를 확인한다.
- 각 단계는 앞 단계 체크포인트 전체를 `model.init`으로 받는다. 1.5 원본을 받으면 LiDAR ViT를 카메라 ViT에서 복사한다. 로드는 strict다: LiDAR 키 말고 빠지거나 남는 파라미터가 있으면 멈춘다. 릴리스에 저장되지 않은 버퍼(Fourier 주파수, (a, κ) 통계)는 config에서 다시 만들고 값을 검사한다.
- 원본 config의 `vlm_name_or_path`(게이트된 `nvidia/Cosmos-Reason2-8B`)를 공개 `Qwen/Qwen3-VL-8B-Instruct`로 바꾼다(recipe 1단계와 같음). 두 토크나이저가 같은 것을 확인했다(151,669, 어휘 일치). 궤적 토큰 id도 원본과 같은지 로드 때 검사한다.

## 로컬 검증 (winpc RTX 5080, 2026-10-24)

환경: `~/claude-bench/venv-teacher`(새로 만듦, `setup_env.sh`와 같은 명령, deepspeed만 뺌), 코드 `~/morai-teacher/train`, 데이터 `D:\share\2026_MORAI_step0`(DrvFs). 로그는 `logs/`에 옮겨 두었다.

**테스트**: `python -m pytest -q tests` → 17 passed.

**확인하지 않은 것**: 실제 10B의 학습(16 GB에 올라가지 않음), DeepSpeed ZeRO-2와 torchrun 실행(winpc에 nvcc가 없음), flash-attention 2(sdpa로만 돌림), Stage 1의 CPU 오프로드 설정. 모두 단계 B 런북에서 처음 돈다.

**데이터 파이프라인** (`logs/sample_cams.jpg`, `logs/sample_bev.png`)
- LiDAR 재포장 1,357개 에피소드 완료(`lidar.bin` 합계 16.4 GB). 아래 학습 검증은 GT 자세로 만든 그 BEV로 했다. 그 뒤 vast 세션이 1,357개 모두를 적분 자세 BEV로 다시 만들었고(실패 0), 새 파일로 200샘플을 다시 읽어 확인했다: 로드·토큰화 정상, 프레임 중복 0, 왕복 오차 평균 0.005 m, 정적 구조물에서 새·옛 스캔이 겹친다(`logs/sample_bev_odom.png`, `logs/sample_cams_odom.jpg`).
- 학습 분할 전체 초기화 131 s(D:, 1,275개 에피소드). 샘플당 디코드 110–174 ms, 토큰화 포함 185–257 ms(단일 프로세스, D: 경합 여부에 따라).
- 200샘플: 프레임 중복 0, (a, κ) 왕복 오차 평균 0.004 m, p95 0.010 m, 최대 FDE 0.064 m.

**1. 같은 구조의 무작위 초기화 소형 모델로 20샘플 과적합** (`configs/tiny_overfit.yaml`, 최종 코드로 다시 돌린 값)
- 실행: `python -m morai_vla.train --config-name tiny_overfit paths.alpamayo15=<릴리스> data.root=<공유>` (릴리스는 config.json만 읽는다).
- 구조: 카메라 ViT + LiDAR ViT(각각 merger·DeepStack 3개), Qwen3-VL LLM, LLM KV를 읽는 expert. 크기만 줄였다(88M, 대부분 155,697×256 임베딩). 학습 부위 전체, 손실 3개(궤적 CE, CoC LM, FM). 데이터는 Stage 1-C 모드(`coc: true`)의 CoC 키프레임 20개, LiDAR dropout 0.15, 배치 2(패딩 경로 포함).
- 2,000 step (`logs/overfit_coc20_2000_loss_components.jsonl`, 963 s, 0.4 s/step, 최대 5.0 GB):

  | step | traj CE | CoC LM | FM |
  |---|---|---|---|
  | 10 | 11.88 | 11.30 | 3.63 |
  | 1000 | 0.0049 | 0.0019 | 1.17 |
  | 2000 | **0.0021** | **0.00085** | **1.06** |

  같은 시점의 처음 보는 val 8샘플은 CE 9.78, LM 3.48, FM 6.78이라 외운 것이 맞다.
- **CE 두 개는 0 근처까지 갔고, FM은 1.06에서 멈췄다.** 이전 실행에서 시각 t별로 나눠 본 결과(`logs/fm_by_t.txt`, 같은 모델 코드)와 같은 현상이다. expert는 맥락만으로 정답 행동 x를 거의 맞히고(expert를 hidden 256으로 키운 실행은 MSE(pred, x) 0.04), 남은 FM ≈ 1은 노이즈 분산, 즉 입력 x_t에서 노이즈를 읽어 내는 부분이다. 무작위 초기화 expert는 Fourier 인코딩을 거친 x_t를 읽는 법을 늦게 배운다. 실제 1.5 expert는 이미 학습돼 있으므로, 실제 가중치로는 런북 3단계(Stage 2 과적합)에서 FM이 0 근처로 가는지 확인한다.
- 이전 실행(키프레임 10 + 일반 10)에서 expert로 궤적을 뽑았을 때, CoC 없는 일반 행은 ADE 1.14 m였다. 키프레임 행은 7.6 m였는데, CoC가 든 프리픽스로 FM을 배우고 CoC 없이 추론해서 생긴 형식 차이다. Stage 2는 CoC 없는 프리픽스로 expert를 학습하므로 이 차이가 없다.
- 체크포인트를 `TeacherVLA.build`로 다시 읽으면 strict 통과(빠진 파라미터 0). CoC를 생성시키면 학습 문장을 그대로 낸다(GT "Turn right to exit the roundabout." → 생성 같음).
- **체크포인트 재개**: 1000 step 체크포인트를 `resume=true`로 이어 돌리면 1010 step부터 원래 실행과 같은 손실을 낸다(`logs/resume_from1000_loss_components.jsonl`; 1010 step traj CE 0.00478 vs 0.00478, LM 0.00177 vs 0.00177, FM은 노이즈 추출 차이 범위 1.158 vs 1.162).
- **실행별 경로** (`logs/stage_paths.txt`): 같은 무작위 초기값에서 step 2·Stage 1-T·1-C·2의 학습 부위, 손실, 데이터 모드, gradient checkpointing(Stage 1류), 카메라 ViT 0.1× 배수로 30 step씩 돌린 뒤 부위별 최대 가중치 변화를 쟀다.

  | 실행 | LiDAR | 카메라 ViT | LLM | LM head | expert |
  |---|---|---|---|---|---|
  | step 2 | 0.0077 | 0 | 0 | 0 | 0 |
  | Stage 1-T | 0.0060 | 0.0006 | 0.0083 | 0.0082 | 0 |
  | Stage 1-C | 0.0056 | 0.0006 | 0.0083 | 0.0083 | 0 |
  | Stage 2 | 0 | 0 | 0 | 0 | 0.0066 |

  카메라 ViT 변화가 LLM의 약 1/10이라 recipe의 0.1배 학습률도 적용된다.

**2. 실제 Alpamayo-1.5-10B, 우리 processor로 MORAI 추론** (`scripts/check_release.py`, `logs/release_nf4/`)
- 계획 변경 전 코드로 잰 값이다. 모델 로드·processor·추론 경로는 그 뒤 바뀌지 않았고, 데이터셋의 샘플 고르는 방식만 바뀌어 다시 돌리면 다른 24샘플이 뽑힌다.
- winpc HF 캐시의 릴리스(스냅샷 `7aba829`)를 썼다. 게이트 승인은 이미 돼 있었다. LLM·expert 선형층은 NF4, ViT·임베딩·LM head·action 투영은 BF16. 로드 136 s, GPU 9.4 GB.
- 로드 검사: 빠진 파라미터 0, LiDAR 키 351개는 카메라 ViT에서 복사(일치 assert), config에서 다시 만드는 버퍼 7개, unexpected 0, mismatched 0.
- val·val_asset 24샘플(정지 대기 샘플 다수 포함), 6.4 s ADE(m):

  | 등속 기준 | 우리 형식, LiDAR 있음 (4샘플 평균 / min) | LiDAR 없음 | CoC 롤아웃(1.5 방식) |
  |---|---|---|---|
  | 7.83 | 8.48 / 5.73 | 8.46 | 8.18 |

- 판단: 출력은 정상 범위다. 정지 샘플은 거의 멈춰 있고(ADE 1.4 안팎), 움직이는 샘플은 이력 속도를 이어 간다. CoC도 장면을 읽는다(커브에서 "Adapt speed for the right curve since the lane bends right ahead"). 그러나 **커브·교차로 회전을 따라가지 못하고 거의 직진**해서 등속보다 낫지 않다. LiDAR ViT는 아직 카메라 사본이라 LiDAR 유무 차이가 없는 것이 맞다.
- 이것이 우리 코드 문제가 아님을 확인한 근거:
  - 원본 1.5 입력 형식(`frame k`, 1.5 카메라 3대, 1.5 픽셀 예산)으로 같은 10샘플을 돌려도 ADE가 같다(우리 7.31, 원본 7.14; `logs/diag_native_prompt.jsonl`).
  - 커브 중간 3샘플(지난 1.5 s 동안 −11~−15° 회전)에서 expert를 BF16으로 바꿔도, upstream 공식 CoC 롤아웃 경로로 돌려도 똑같이 직진한다(`logs/diag_context_probe_*.jsonl`).
  - 이 PC의 이전 프로젝트(Alpamayo-demo-MORAI)가 공식 alpamayo1.5 코드로 MORAI bag 1,523프레임을 돌렸을 때도 minADE(6.4 s, K=6) 22.6 m였다.
  - 결론: 미세조정 전 1.5의 MORAI 도메인 한계다. step 2·3에서 고칠 대상이다.

**이 과정에서 고친 것** (모두 위 결정에 반영)
- 프레임 중복(9.9% 행)과 미래 궤적 지터(위치 대신 속도·yaw 적분), 속도 글리치 행 제외.
- 원본 체크포인트에 없는 정규화 버퍼가 BF16 로드 때 반올림되던 것: 로드 뒤 `action_space`를 config에서 fp32로 다시 만들고 값을 assert한다.
- 계획 변경(공식 recipe 기준)에 맞춰 혼합 CoC 비율·reasoning dropout·step 2의 FM 손실을 없애고, CoC를 인덱스에서 빼 설정한 parquet에서 읽게 했다.
- HF Trainer가 `**kwargs`를 받는 모델은 손실을 그래디언트 누적 수로 나누지 않으므로 `accepts_loss_kwargs = False`.
- 추론은 upstream처럼 BF16 autocast 안에서 돈다(그렇지 않으면 action_in_proj dtype 오류).

## 단계 B 런북: vast.ai 1× H100 80 GB

목표: 실제 가중치 strict 로드 → 20샘플 과적합 → 메모리·배치 측정 → IO 처리량 → 체크포인트 재개 → step 2 완주. Stage 1-T·1-C는 GPU 8장이 필요해서(위 단계 표) 단계 C에서 돈다.

### 0. 준비
- 인스턴스·데이터·체크포인트·venv는 [VAST_SETUP.md](../VAST_SETUP.md) 2.2절, 3.5절, 4.3절, 5절을 따른다. 경로는 그 문서와 같다: 데이터 `/data/step0`(에피소드마다 `cams.bin, cams_index.npz, lidar.bin, lidar_index.npz` + `_ego/`, `_index/`, `_coc/`), CoC v1.1 `/data/coc/coc_v1_1/coc_dataset.parquet`, 릴리스 `/workspace/ckpt/Alpamayo-1.5-10B`(A1 변환본은 쓰지 않는다), 결과 `/workspace/out`, `HF_HOME=/workspace/hf`.
- venv는 recipe의 `a1_5_sft`(5절, `uv sync --active`)를 그대로 쓴다. 이 코드가 쓰는 패키지는 모두 그 lock에 있다(bitsandbytes만 없는데 16 GB 카드용 NF4 점검에만 쓴다). winpc에서는 같은 버전을 `setup_env.sh`로 깔았다.
- 이 `train/` 폴더를 `/workspace/train`에 둔다(Mac에서 `rsync -a --exclude logs train/ <인스턴스>:/workspace/train/`).
- 셸 준비 (모든 명령은 이 셸에서):
  ```bash
  source /workspace/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
  cd /workspace/train
  export PYTHONPATH=/workspace/train HF_HOME=/workspace/hf OUT=/workspace/out
  export C="paths.alpamayo15=/workspace/ckpt/Alpamayo-1.5-10B data.root=/data/step0 data.coc_parquet=/data/coc/coc_v1_1/coc_dataset.parquet"
  export T="torchrun --nproc_per_node 1 -m morai_vla.train"
  ```
  v1.1이 아직 없으면 `data.coc_parquet=...`를 빼면 `/data/step0/_coc`의 v1을 쓴다. CoC는 Stage 1-C에서만 읽는다.
- 학습은 recipe처럼 `torchrun`으로 띄운다(`$T`). GPU 1장이어도 DeepSpeed(ZeRO-2)가 분산 환경 변수를 요구하기 때문이다.

### 1. 테스트
```bash
python -m pytest -q tests          # 23 passed
```

### 2. 실제 가중치 strict 로드와 추론 점검
```bash
python scripts/check_release.py --release /workspace/ckpt/Alpamayo-1.5-10B --root /data/step0 --no-quant --n 48 --out $OUT/release_bf16
```
- 로그에 `missing params 0 (lidar 351), config-derived buffers 7, unexpected 0, mismatched 0`이 나와야 한다(winpc와 같음). 다른 파라미터가 빠지면 `strict`가 예외를 낸다. LiDAR ViT가 카메라 ViT 사본인지, 정규화 통계가 config와 같은지는 코드가 assert한다.
- `summary.json`의 traj ADE를 winpc NF4 값(위 로컬 검증, 8.48 m / 등속 7.83 m)과 비교한다. BF16이 같거나 낮아야 한다.

### 3. 20샘플 과적합 (실제 가중치)
```bash
O="data.train_dataset.max_samples=20 data.train_dataset.augment=false trainer.max_steps=300 trainer.gradient_accumulation_steps=1 trainer.warmup_steps=10 trainer.eval_strategy=no trainer.save_strategy=no"
$T --config-name step2  $C paths.output_dir=$OUT/ovf_step2  $O
$T --config-name stage2 $C model.init=/workspace/ckpt/Alpamayo-1.5-10B paths.output_dir=$OUT/ovf_stage2 $O
```
- 손실은 `$OUT/*/loss_components.jsonl`에서 본다.
- step 2는 LiDAR 경로만 학습하므로 0까지 가지 않아도 되고, `traj_ce`가 꾸준히 내려가면 된다.
- Stage 2는 이미 학습된 1.5 expert라 `fm`이 0 근처까지 내려가야 한다(소형 무작위 expert와 다른 점, 위 로컬 검증).
- recipe의 20샘플 스모크(Stage 1, VLM 전체)는 학습 파라미터 9.5B라 1장에서는 `trainer.deepspeed=configs/deepspeed/zero2_offload.json`로만 된다(CPU에서 AdamW, CPU RAM 200 GB 이상, 느림, 이 코드로 돌려 본 적 없음). 선택 사항이다:
  ```bash
  $T --config-name stage1t $C model.init=/workspace/ckpt/Alpamayo-1.5-10B paths.output_dir=$OUT/ovf_stage1t trainer.deepspeed=configs/deepspeed/zero2_offload.json $O
  ```

### 4. 메모리·배치 측정
`loss_components.jsonl`에 `peak_mem_gb`, `s_per_step`이 찍힌다.
```bash
P="trainer.max_steps=30 trainer.logging_steps=10 trainer.gradient_accumulation_steps=1 trainer.eval_strategy=no trainer.save_strategy=no"
for B in 1 2 4; do $T --config-name step2 $C paths.output_dir=$OUT/prof_step2_b$B trainer.per_device_train_batch_size=$B $P; done
for B in 1 2; do $T --config-name stage2 $C model.init=/workspace/ckpt/Alpamayo-1.5-10B paths.output_dir=$OUT/prof_stage2_b$B trainer.per_device_train_batch_size=$B $P $( [ $B -gt 1 ] && echo model.attn_implementation=sdpa ); done
```
- step 2(VLM만 통과, gradient checkpointing)는 배치를 늘려도 flash-attention 2 그대로 된다.
- Stage 2는 배치 2 이상이면 프리픽스에 왼쪽 패딩이 생긴다. expert 패딩 마스크는 sdpa/eager에서만 되므로 그때만 sdpa로 잰다(flash-attention 2면 일부러 예외를 낸다).
- 결과로 각 실행의 `per_device_train_batch_size`·`gradient_accumulation_steps`를 정하고(recipe 유효 배치: Stage 1 = 4 × GPU 수, Stage 2 = 1 × GPU 수), 단계 C의 시간 추정(VAST_SETUP 2.3절)을 이 `s_per_step`으로 바꾼다.

### 5. IO 처리량
```bash
python scripts/bench_io.py --config-name step2 $C trainer.dataloader_num_workers=8 +n=400
```
- 출력 samples/s가 4단계의 GPU 처리량(배치 ÷ `s_per_step`)의 1.5배 이상이어야 한다. 모자라면 workers를 늘린다(단계 C는 GPU 8장 × workers).
- winpc(D: DrvFs, 단일 프로세스)에서는 토큰화까지 샘플당 185–257 ms, 전체 train 분할 초기화는 131 s였다.

### 6. 체크포인트 재개
```bash
R="trainer.max_steps=40 trainer.save_steps=20 trainer.warmup_steps=5 trainer.eval_strategy=no"
$T --config-name step2 $C paths.output_dir=$OUT/resume $R          # checkpoint-20이 생긴 뒤 Ctrl-C
$T --config-name step2 $C paths.output_dir=$OUT/resume $R resume=true
```
- 재개한 실행의 첫 손실 줄이 step 30(로깅 10 step 단위)부터 나오고, 값이 중단 전 추세와 이어져야 한다. 같은 절차를 소형 모델로 winpc에서 확인했다(위 로컬 검증).
- checkpoint 하나가 약 71 GB다(HF 가중치 22 + DeepSpeed model states 43 + optimizer 6.5). 데이터와 같은 디스크라면 여유를 먼저 확인한다(단계 B 결과).

### 7. step 2 본 실행
```bash
nohup $T --config-name step2 $C paths.output_dir=$OUT/step2 trainer.per_device_train_batch_size=4 trainer.gradient_accumulation_steps=1 trainer.save_strategy=no > $OUT/step2.log 2>&1 &
```
- recipe Stage 1 값 그대로다: lr 1e-5, warmup 500, 유효 배치 4, ZeRO-2, gradient checkpointing. LiDAR 경로만 학습하고 2000 optimizer step(8k 샘플)이다. 500 step마다 val 손실(`val`+`val_asset` 512샘플)을 잰다.
- 배치 4 × 누적 1은 1 × 4와 유효 배치가 같고 샘플당 약 25 % 빠르다(4단계 측정). 중간 저장을 끈 것은 checkpoint 하나가 71 GB라서다. 디스크가 넉넉하면 `save_strategy=no`를 빼면 500 step마다 저장한다.
- 짧은 예열이 목적이다(MODEL_PLAN 4.2). 중간 저장이 없으므로 끊기면 처음부터 다시 돈다(H100 1장 79분). 저장을 켰다면 같은 명령에 `resume=true`.
- 결과 `$OUT/step2/final`(또는 `checkpoint-N`)이 Stage 1-T의 `model.init`이다. 인스턴스를 지우기 전에 `$OUT`을 B2에 올린다(VAST_SETUP 3.6절).

### 그다음 (단계 C, GPU 8장)
아래 세 명령을 손으로 돌리는 대신, 판정·B2 저장·재개까지 포함한 `scripts/pipeline.sh`가 같은 순서로 돈다(아래 "단계 C 파이프라인").
```bash
torchrun --nproc_per_node 8 -m morai_vla.train --config-name stage1t $C model.init=<step2 결과> paths.output_dir=$OUT/stage1t
torchrun --nproc_per_node 8 -m morai_vla.train --config-name stage1c $C model.init=<1-T 결과> paths.output_dir=$OUT/stage1c
torchrun --nproc_per_node 8 -m morai_vla.train --config-name stage2  $C model.init=<1-C 결과> paths.output_dir=$OUT/stage2
```

### Stage 1 모니터링 (expert 없이)

step 2 결과에서 보듯, VLM이 움직이는 동안 고정 expert를 거친 궤적(`check_release`의 `traj`)은 Stage 2 전까지 지표가 아니다. teacher-forced CE만 보면 자유 생성이 나쁜 것을 놓칠 수 있어서, Stage 1-T·1-C는 eval마다 `morai_vla/stage1_eval.py`(`Stage1EvalCallback`, `configs/stage1t.yaml`·`stage1c.yaml`)를 돌린다(기록만, 학습·선택에 쓰지 않는다).
- 샘플: check_release 1,000샘플 집합의 앞 200개(같은 seed·stride, 그래서 BF16 기준값과 샘플 단위로 맞는다). eval 간격마다, rank별로 나눠 계산하고 rank 0이 `<output_dir>/stage1_eval.jsonl`에 쓴다.
- `ce_on` / `ce_off`: 같은 샘플의 teacher-forced 궤적 CE, LiDAR 있음 / 강제 제거. `ce_on`만 내려가고 `ce_off`가 멈추면 LiDAR dropout(0.15)을 올린다.
- `ade` / `fde`: 배포 프리픽스(CoC 없음)에서 궤적 bin으로 제한한 greedy 생성 128토큰 → 모델의 `DiscreteTrajectoryTokenizer`로 (a, κ) → 적분, 6.4 s. `cv_ade` / `cv_fde`는 같은 샘플의 등속 기준. 셀별 값도 쓴다.
- 단독 실행: `python scripts/eval_stage1.py <체크포인트> <step-0 루트> --n 200 [--nf4] --out x.json`.
- 시작 기준값 (winpc 5080 NF4, 1,000샘플 집합의 앞 50개, `logs/stage1_eval/`). NF4 CE는 같은 샘플의 BF16 CE와 평균 0.014 차, 상관 1.000이라 왜곡이 작다. 본 실행은 t1t 시작 때(`eval_on_start`) step 2를 BF16 200샘플로 다시 잰다.

  | | ce_on | ce_off | 이산 토큰 ADE / FDE | expert ADE (BF16) | 등속 ADE / FDE |
  |---|---|---|---|---|---|
  | 1.5 release | 1.371 | 1.382 | 3.92 / 12.05 | 5.09 | 5.16 / 12.99 |
  | step 2 | 1.114 | 1.382 | 3.90 / 12.10 | 8.53 | 5.16 / 12.99 |

  이산 토큰 궤적은 이미 등속과 expert 경로보다 낫다. step 2의 LiDAR는 CE를 0.26 낮췄지만 greedy 궤적 ADE는 그대로다(3.92 → 3.90, 50개 중 24개 개선).

토크나이저 범위: `scripts/tokenizer_clip.py`. train 2 Hz 364,140궤적에서 [-10, 10] 밖으로 잘리는 미래 토큰은 0.33 %(가속도 토큰 0.67 %, 곡률 0.0001 %), 잘린 토큰이 하나라도 있는 궤적 6.8 %. 정규화 가속도 표준편차 2.09(PAI 약 1), |a|/σ 99.9 분위 14.7(약 10 m/s²). 곡률은 문제없고, 가속도는 급제동 구간의 끝값만 잘린다(보고만 한다).

## 단계 C 파이프라인 (8× B200 한 번 대여로 최종 모델까지, 10/29까지)

`scripts/pipeline.sh` 하나가 무인으로 끝까지 돈다. 교사 단계는 이 파일에, 학생 단계(soft target부터 최종 내보내기까지)는 학생 코드가 맡는 `scripts/pipeline_student.sh`의 함수로 있다.

```bash
setsid nohup bash scripts/pipeline.sh > /workspace/out/pipeline.log 2>&1 < /dev/null &
STAGES="t1t" T1T_EPOCHS=1 bash scripts/pipeline.sh       # 일부 단계만 (시험)
STAGES=smoke8 OUT=/workspace/out/smoke bash scripts/pipeline.sh   # 새 상자의 첫 몇 분: 전 GPU NCCL/ZeRO-2, 실제 설정의 메모리, Stage 1 모니터, B2 저장
```
smoke8의 `loss_components.jsonl` `peak_mem_gb`가 GPU 메모리의 90 %를 넘으면(H200 141 GB에서 약 127 GB) `T1_GC=true`로 돌린다(GC 켬: 1 GPU 40 GB, 약 2배 느림).
경로는 `W`(기본 `/workspace`), `VENV`, `TRAIN`, `REL`, `STEP2`, `ROOT`, `COCP`, `OUT`, `B2` 환경 변수로 바꾼다(RunPod 등 다른 배치).
준비물: VAST_SETUP 5절의 venv, 전체 데이터 `/data/step0` + `/data/coc/coc_v1_1`, `/workspace/ckpt/Alpamayo-1.5-10B`, `/workspace/ckpt/step2_final`(B2 `runs/stageB/step2/final`), PATH의 `b2` CLI와 `/workspace/b2.env`(`B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`), `tools/vast/idle_watchdog.sh`.

| 단계 | 입력 → 출력 (`$OUT=/workspace/out/stageC`) | 내용 |
|---|---|---|
| `t1t` | step2_final → `t1t/final` | Stage 1-T, 3 epoch 스케줄을 checkpoint-`T1T_STOP`(16000)까지 + 1,500 step anneal, GPU당 배치 4 × 누적 1, GC 끔, eval·저장 2000 step마다(+ Stage 1 모니터), `final/` = 세 후보 중 이산 토큰 ADE 최저 |
| `t1c` | t1t → `t1c/final` | Stage 1-C, 3 epoch, eval·저장 200 step마다(+ Stage 1 모니터), `final/` = eval_loss 최선 |
| `t2` | t1c → `t2/final` = T* | Stage 2, 3 epoch 상한, GPU당 배치 1, DeepSpeed 없음, eval·저장 2000 step마다(재개용). 4,000 step마다 궤적 ADE 측정, 규칙대로 멈추면 3,000 step 선형 감쇠, `final/` = ADE 최저 측정(`morai_vla/early_stop.py`, `ade_select.json`) |
| `tgate` | T* → `tgate/summary.json` | 1,000 val 샘플 check_release(LiDAR 있음/없음, GPU마다 1/N씩) + Stage 1 모니터 |
| `soft`, `s1`, `sgate`, `s2`, `qad`, `deliver` | `pipeline_student.sh` | soft target + 규칙 검사, 학생 모델 와우마요(WoWmayo) 1단계, 학생 관문, 2단계, QAD, 5080용 최종 내보내기 |

- 단계마다: 완료 표시가 있으면 건너뛴다 → 실행 → 자동 판정 → `$OUT/<단계>`를 B2 `runs/stageC/<단계>/`에 올린다(`checkpoint-*` 제외) → 파일 수·바이트를 B2 목록과 비교 → 완료 표시를 로컬과 B2 `runs/stageC/done/`에 쓴다. 그다음 그 단계의 `checkpoint-*`(재개용, 하나에 100–160 GB)와 `ade_snapshots/`(측정마다 남긴 학습 대상 가중치, B2에 올리지 않음)를 지운다. 단계마다 최선+최신이 남으면 1.5 TB 볼륨을 넘기 때문이다.
- 실패나 판정 탈락이면 `$OUT/HALT`에 이유를 쓰고 로그를 올린 뒤 멈춘다. GPU가 놀면 watchdog이 인스턴스를 멈춘다.
- 다시 실행하면 마지막 완료 단계 다음부터 이어 간다. 같은 상자면 단계 안 체크포인트에서 `resume=true`로, 새 상자면 필요한 입력 `final/`을 B2에서 받는다. 학습 중 체크포인트는 단계마다 하나만 둔다(9.4B ZeRO-2 하나가 약 150 GB).

**자동 정지 기준** (실행 전에 정한 값, `scripts/pipeline_gate.py`, `morai_vla/guards.py`)

| 대상 | 정지 조건 |
|---|---|
| 모든 학습 단계 | 손실이나 grad norm이 처음 NaN/inf가 되는 순간 학습을 끝내고(`NonFiniteStop`) 파이프라인 정지 |
| t1t | NaN, 또는 고른 이산 토큰 ADE가 같은 1,000샘플의 등속 ADE보다 낮지 않음 (CE 발산 정지 없음) |
| t2, s1, s2 | 마지막 eval 손실(항목별, ~0인 값 제외)이 그 실행 최저값의 1.2배 초과 |
| t1c | NaN만 (CoC를 빨리 외워 eval 손실이 오르는 것은 정지 사유로 보지 않는다) |
| tgate | T* expert 6.4 s ADE(LiDAR 있음, 1,000 val 샘플, 6개 샘플 평균)가 1.5 release의 5.29 m 미만이 아니면 |
| sgate | 학생(고정 expert) ADE ÷ T* ADE(`gate_student.py` summary의 `ratio`, 같은 샘플·같은 초기 노이즈)가 1.5 이상. 계획의 목표 1.1 이하는 보고만 한다(못 미치면 PAI 사전 증류를 정할 일, "학생" 절 런북 6) |
| qad | 양자화 학생 ADE가 s2 학생 ADE의 1.25배 초과, 또는 NaN (학생 코드가 확정) |

그 밖의 값(셀별 ADE, FDE, minADE, LiDAR 없는 ADE, CE on/off, 이산 토큰 ADE, 학습 시간)은 기록하고 보고만 한다.

**예산**: 아래 "B200 최적화와 단계 C 시간"을 본다.

### 단계 C 운영 기록 (8× H200, RunPod xfvpbway4kj26i, 10/25~)

- 08:41 smoke8 통과: 8 GPU ZeRO-2, GPU당 배치 4·GC 끔 메모리 121.5 GB(141 GB 중), Stage 1 모니터, B2 업로드·확인.
- 08:47 t1t 시작 → 09:06 HALT. 두 가지가 겹쳤다.
  1. **LiDAR dropout과 ZeRO-2 멈춤**: 한 rank의 마이크로배치 샘플이 모두 LiDAR 빠짐이면 그 rank만 `lidar_visual`을 건너뛰어 그래디언트 훅 순서가 rank마다 달라지고 all-reduce가 멈춘다(배치 4: rank-step당 0.15⁴, 55 step에서; 배치 2: step 0에서). `LidarQwen3VLModel.get_image_features`가 학습 중 그런 배치에 0을 곱한 LiDAR ViT 통과를 넣어 모든 rank의 그래프를 같게 한다(`tests/test_lidar_grad.py`). 학생도 같은 클래스를 쓴다.
  2. **GPU 3 과열**: 부하 시 88 °C에서 SM 480 MHz(정상 1980), 쉴 때도 82 °C. 동기 학습이 가장 느린 GPU에 맞춰져 4.8 s/step(예상의 약 3배). `GPUS=0,1,2,4,5,6,7`로 7장만 쓴다(전체 배치 28, recipe 32).
  - NCCL NVLS도 의심해 기본값으로 끈다(`NCCL_NVLS_ENABLE=0`, all-reduce 350 GB/s, 켜면 452 GB/s). 데이터로더는 MooseFS에서 워커 8개 60 samples/s로 병목이 아니다.
- 09:28 7장으로 t1t 재시작: 1.54 s/step(GPU당 0.39 s/샘플), 2 epoch 19,218 step ≈ 8.5 h.
- 09:31 조기 종료(`EarlyStoppingCallback`)와 최선 체크포인트 선택을 넣고 t1t를 재시작했다(다른 세션의 요청을 사용자 요청으로 잘못 받음). 그 뒤 셀 균형 val 지표도 만들었다.
- **사용자 결정으로 되돌림** ("공식 레시피대로 하고 손대지 말자", "니가 영향끼친거 다 지워" = 09:20 이후 btw 세션이 넣은 것): 조기 종료, 최선 체크포인트 선택(`final/` = 마지막 체크포인트, recipe), 셀 균형 val(`val_sets.py` 삭제)을 뺐다. eval·저장 간격도 그 전으로(t1t eval 1000·저장 2000, t2 eval 500·저장 2000, `save_total_limit 1`). 그 전부터 있던 것(메인 세션 요청)은 그대로: Stage 1 모니터 콜백과 `eval_on_start`, 속도 설정(GPU당 배치 4 × 누적 1, GC 끔), NCCL·LiDAR 멈춤 수정, 파이프라인, 학생 연결.
- **t1t epoch = 3 (recipe)**: 사용자 "3에폭으로하면? 이거 속도 개선 안됄때 줄이기로 한거였잖아". 2 epoch는 3 s/step을 넘을 때의 상한이었고, 1.54 s/step이면 3 epoch(28,827 step, 약 12.3 h + eval)가 들어간다.
- **최선 체크포인트 사용** (사용자 "베스트 체크포인트 쓰는 걸로 하자"): t1t·t1c·t2에 `load_best_model_at_end` + `metric_for_best_model: eval_loss`(recipe의 val 512). 조기 종료는 없다(epoch 끝까지). HF는 저장한 체크포인트 중에서만 고르므로 eval = 저장 간격(t1t·t2 2000, t1c 200). 최선은 `save_total_limit`에 지워지지 않고, 고른 step과 값은 `<stage>/best.json`(B2에도 올라감). 디스크: t1t 체크포인트 약 155 GB × (최선 + 최신 + 교체 중 1) ≈ 465 GB, 1.5 TB 볼륨에 데이터 203 GB.
- 10:31 checkpoint-2000 저장(153 GB, 약 4분). step 2000 val traj CE 1.043(step 0: 1.172), Stage 1 모니터 CE LiDAR 있음/없음 1.100/1.104, 이산 ADE 2.92 m(등속 5.21).
- 10:32 이 설정으로 checkpoint-2000에서 `resume`(`/workspace/switch_at_2000.sh`): 콜백 loss_components·nonfinite_stop·stage1_eval, 3 epoch = 28,827 step(cosine 길이 재설정), load_best_model_at_end + eval_loss, eval·저장 2000, save_total_limit 1. step 2000부터 이어서 1.54 s/step. 학습 손실 없음.
- **t1t를 checkpoint-16000에서 멈추고 anneal 후 ADE로 선택**(사용자 결정, 16:55): eval traj CE 1.043(2000) → 1.052 → 1.084 → 1.132 → 1.198 → 1.256(12000)로 오르는 동안 이산 토큰 ADE는 2.92 → 2.74 → 1.94 → 1.73(8000) → 1.80 → 1.82 m(등속 5.21). eval_loss 최선(2000)은 궤적이 가장 나쁘고, CE 발산 정지(1.2배)도 걸린다. 그래서 `pipeline.sh`의 t1t: checkpoint-`T1T_STOP`(16000) 뒤 `/workspace/stop_at_16000.sh`가 본 실행을 멈추고 파이프라인을 다시 띄운다 → `t1t_anneal`(16000에서 재개, `T1T_ANNEAL`=1,500 step, LR 그 시점 값에서 0까지 선형) → `t1t_select`(step-12000 스냅샷·checkpoint-16000·anneal을 1,000샘플 이산 ADE로, GPU 7장에 나눠) → 가장 낮은 것이 `t1t/final`, 세 후보는 `t1t/best.json`. t1t 관문은 NaN과 'ADE < 등속'만. checkpoint-16000은 처음에 정리 대상에서 뺐다가, 새 pod 이전 때 사용자 결정으로 버린다(B2에 올리지 않음, 옛 pod와 함께 삭제).
- 체크포인트 보존: `t1t_snapshots/step-N`(HF 가중치 22 GB)에 2000·12000·14000·16000을 둔다. 4000–10000은 이미 교체돼 없다.
- 16:07 RunPod 잔액이 $10.89까지 떨어져 checkpoint-12000·2000을 B2 `runs/stageC/t1t_ckpt/`에 올리기 시작했다(16:19에 사용자가 충전, 정지 없음). B2가 병렬 대용량 업로드에 429 TooManyRequests를 내서 파이프라인 업로드는 스레드 8개·3번 재시도로 바꿨다.
- **t1t 결과 (18:38)**: anneal 1,500 step(LR 4.81e-7 → 0, 첫 그룹 기준) 뒤 1,000샘플 이산 토큰 ADE / FDE(m), CE LiDAR 있음/없음: step-12000 1.488 / 4.474, 1.261 / 1.271; checkpoint-16000 1.591 / 4.671, 1.320 / 1.331; **anneal 1.324 / 3.997**, 1.403 / 1.415 → `t1t/final` = anneal. 등속 5.143 / 12.765. CE는 여전히 오르는데 궤적은 anneal이 가장 좋다. t1t 1.33 h(재시작 이후), B2 `runs/stageC/t1t/`.
- **t1t 횡/종 오차** (사용자 질문 "차선 다 밟는거 아님?", `scripts/lateral_error.py`, 같은 1,000샘플, GT 진행 방향 기준, `logs/stageC/lateral.json`): ADE 1.32 m의 대부분은 종방향(속도·시점)이다. |횡| 평균/p90: 1 s 0.02/0.05 m, 2 s 0.07/0.16 m, 3 s 0.13/0.31 m, 6.4 s 0.51/1.35 m; 종 평균 2 s 0.33 m, 6.4 s 4.03 m. 처음 2 s에 |횡| > 0.5 m인 샘플 3.0 %, > 1.0 m 0.8 %. 회전(6.4 s 동안 15° 넘게 꺾임) 185개는 3 s 0.22/0.57 m, 6.4 s 1.00/2.52 m. 셀 C5가 가장 나쁘다(2 s 0.20 m, > 0.5 m 10 %). 등속은 2 s 0.71 m, 6.4 s 4.49 m, > 0.5 m 29 %.
- **C5 이상치** (`logs/stageC/c5_outliers.png`, 처음 2 s |횡| > 0.5 m인 C5 7개 + 전체 최악 3개): 대부분 고속도로 차로 변경 판단(바꿀지·어느 쪽·언제) 차이다. 공사 구간에서 GT는 오른쪽으로 비키는데 예측은 유지(run1020t-w2-007/4730), 같은 장면에서 변경 방향이 반대(run1021v-w1-043/950), 합류 화살표에서 늦은 변경, 안개 속 끼어드는 차 옆에서 짧게 감, 톨게이트 차로 선택. 정상 주행 중 차선 이탈은 아니다.
- **신호 판독** (`scripts/signal_eval.py`, `logs/stageC/signal_eval.*`, val+val_asset에서 다음 판정 정지선이 5–80 m 앞인 394행, t0 신호 × 안개 층화, 이산 토큰 greedy 궤적의 앞축이 6.4 s 안에 그 정지선을 넘는지): 전체 일치 94.2 %. 빨강 133개 중 GT 정지 131 → 예측 통과 2(신호 위반률 1.5 %, 둘 다 안개 속 정지선 4188 62–63 m, 카메라에 신호가 거의 안 보임). 초록 일치 89.6 %, GT 통과 115 중 예측 정지 12(40–60 m 맑음에 5). 노랑 일치 96.1 %. 불필요한 정지 17/125 = 13.6 %. 거리별 일치 5–20 m 98.4 %, 20–40 m 93.8 %, 40–60 m 93.1 %, 60–80 m 93.1 %. 안개 94.4 % vs 맑음 93.9 %. 한계: t0 신호만 본다(6.4 s 안의 신호 변화 무시), 60–80 m 저속은 GT도 6.4 s 안에 선에 못 가서 '정지'로 잡힌다, 4356 화살표 구분 없음.
- **t1c 결과 (19:39)**: 1,017 step(3 epoch, 1.01 h). eval_loss(CoC LM 포함) 최선 = step 200(2.472) → `t1c/final`, B2 `runs/stageC/t1c/`(14파일 23.3 GB 확인). eval traj CE는 400 step 2.375 → 1000 step 2.604로 t1t처럼 오른다. 큐는 t2가 일을 시작하기 전에 멈췄다.
  - 주의: `save_total_limit 1` + `load_best_model_at_end`면 HF가 학습 끝에 최선이 아닌 마지막 체크포인트를 지운다. 그래서 t1c의 마지막(1017) 가중치는 없다. t2·학생 단계에서 궤적으로 후보를 다시 고르려면 t1t처럼 스냅샷을 남기거나 `save_total_limit 2`로 둔다.
- **늦은 출발 분해** (`logs/stageC/late_departure.*`, 서 있다가 GT가 2 s 안에 출발한 72건): t0에 이미 초록/노랑 47건(예측이 2 s 그대로 79 %, 지연 중앙값 4.7 s), t0 빨강→이후 초록 10건(예측 불가, 100 %), 빨강→빨강 8건(40 m 대기에서 선으로 다가감), 미확인 7건. 거리: 35–50 m 35건(드라이버가 빨강을 40 m 앞에서 기다린 곳), 50 m 넘음/정지선 없음 26건, 0–8 m 1건. 안개 72 % vs 맑음 87.5 %.
- **정지 수렴** (`logs/stageC/stationary.json`): GT가 6.4 s 내내 서 있는 514개에서 예측 이동 중앙값·p90 0.008 m, 처음 3 s 0.5 m 초과 0.8 %, 1 m 초과 0.2 %(1건 17 m, C6).
- **정지선 6254** (`scripts/cam7_eval.py`, winpc 5080 **NF4**, `logs/stageC/stop6254_nf4.*`, 정지선 8 m 안에서 v < 1 m/s, val+val_asset 151개): GT가 3 s 서 있는 75개(빨강 대기)에서 예측도 100 % 서 있고 빨간불 위반 0, 미확인 신호에서 넘음 0. GT가 2 s 안에 출발하는 72개 중 예측이 2 s 그대로 52개(72 %), 출발 중앙값 GT 1.2 s vs 예측 6.4 s 넘음. 안개·맑음이 같다(각 26/38). 주의: 6254에서는 _ego 신호 코드가 출발 72건에서 t0·출발 시점 모두 미확인 56, 빨강 16이고 초록이 한 번도 없다. 그래서 초록 직후 출발인지 신호로 확인할 수 없고, 위반 판정도 빨강으로 기록된 경우만 본다.
- **t2 재시작 (10/26, vast 57691612 8× H200)**: 02:03 KST 시작, 0.36 s/step, 100,896 step(3 epoch), 메모리 60 GB. 첫 ADE 측정(step 4000)에서 멈췄다. 학습 모델은 flash_attention_2인데, K개 반복 샘플을 한 번에 뽑는 expert 호출이 4D 마스크를 받지 못했다(`cu_seqlens_q must have shape`). winpc 시험은 sdpa여서 드러나지 않았다. 고친 방법: `sample_trajectories`가 궤적을 뽑는 동안만 expert를 sdpa로 돌린다. 실제 t1c 모델로 FA2와 sdpa의 ADE가 같은지 확인하고(샘플당 0.5 s), checkpoint-2000에서 재개했다(02:32).
- **단계 C 완료 (10/27 05:38 KST, vast 57691612, B2 `runs/stageC/done/` 10개)**:
  | 단계 | 결과 | 시간 |
  |---|---|---|
  | t2 | step 60,000에서 조기 종료 → 63,000까지 감쇠. `final/` = step 48,000, ADE 1.307(감쇠 후 1.371). 3 epoch 100,896 step 중 63,000만 돎 | 6.95 h |
  | tgate | T* expert ADE / FDE: LiDAR 있음 1.311 / 3.865, 없음 1.323 / 3.902 (등속 5.143, 릴리스 기준 5.29) | 0.4 h |
  | soft | soft target | 3.48 h |
  | s1 (와우마요 1단계) | step 10,200에서 조기 종료 → 10,700까지 감쇠. `final/` = step 9,600, ADE 1.761 | 7.7 h |
  | sgate | 학생 ADE 1.781 / T* 1.350, 비율 1.319 (기준 1.5 미만) | 0.23 h |
  | s2 | eval_fm 최선 | 6.68 h |
  | qad | 예산 4.75 bits(사용자 10/26)에서 탐색 결과 4.749, 600층, 탐색 175 s. 최선 checkpoint-400(eval_fm 0.259), `ratio_vs_s2` 1.015(QAD 1.511 / s2 1.489, 500샘플), 최대 메모리 78.4 GB. 첫 시도는 시작 21 s 만에 HALT, 재시작 후 통과 | 1.41 h |
  | deliver | `deliver/final` 15.09 GB (5080 엔진 변환 입력) | 0.04 h |
- **궤적 ADE 자동 조기 종료** (10/25 사용자 규칙, `morai_vla/early_stop.py` `TrajEarlyStop`, Stage 2 `configs/stage2.yaml`, 와우마요(WoWmayo) 1단계 공용): 학습 중 `every_steps`마다 모든 학습 GPU가 고정 1,000샘플(check_release·t1t 선택과 같은 집합)을 나눠 expert 연속 궤적 ADE(6.4 s, K=6 평균, LiDAR 켬, 샘플마다 seed 0)를 잰다. 0.5 epoch 뒤로 그때까지 최고보다 2% 넘게 낮아지지 않은 측정이 3번 연속이면 그 step의 스케줄 LR에서 0까지 `decay_steps` 동안 선형으로 내리고(같은 프로세스, LambdaLR의 lambda 교체), 끝에서 한 번 더 잰다. 측정마다 학습 대상 파라미터를 `ade_snapshots/step-N.safetensors`로 남기고, `train.py`가 끝에서 ADE 최저 측정을 모델에 덮어써 `final/`로 저장한다(`ade_select.json`·`ade_early_stop.json`이 B2 기록). 상태가 파일에 있어 재개해도 이어진다. winpc 확인: `configs/tiny_stage2.yaml` + `scripts/es_test.sh`, 단위 테스트 `tests/test_early_stop.py`.
- **정지선 7294·4188·4356** (`scripts/cam7_eval.py`, 6254와 같은 조건: winpc NF4, 정지선 8 m 안에서 v < 1 m/s, 정지선마다 최대 150개, 적으면 전부): 7294는 92개 모두 한 에피소드(run1020s-w2-014)다. 앞 차축이 이미 정지선 위에 선 채 빨강 92/92이고, GT는 출발하지 않는다(go 0). 예측도 0.25 m 안에서 그대로다. 스크립트가 센 "빨간불 위반" 89는 판정 오류다. 서 있는 궤적의 0.1 m 미만 흔들림에서 방향을 구하면 앞 차축 점이 정지선을 왔다 갔다 해서 생긴다(GT 궤적도 5번 걸림). 실제 위반은 0이다. 4188과 4356은 해당 샘플이 0개다. 수집 운전기는 빨간불에 앞 범퍼 기준 7294 16 m, 6254·4188 36 m, 4356 31 m 앞에서 선다. 그래서 서 있는 행이 4188은 35–45 m에 36,716개, 4356은 25–60 m에 1,735개 있고, 8 m 안에는 없다. 6254의 8 m 안 151개도 공사 회피 중 정지선 앞에 묶인 특수 경우다.
- **정지선 4곳, 실제 정지 위치** (사용자 10/25, `logs/stageC/hold*_nf4.*`, winpc NF4, val+val_asset, v < 1 m/s, 정지선마다 최대 150개). 범위는 차량 기준점에서 정지선까지 경로 거리로 7294 10–25 m, 6254·4188·4356 25–45 m이고, 서 있는 행 분포의 몰린 곳과 맞는다. 출발은 2 s 안에 0.5 m 넘게, 6.4 s 동안 5 m 이상 움직인 경우다(조금 다가가기만 한 행은 제외). 늦은 출발은 예측이 2 s 동안 0.5 m 안에 머문 경우다. 위반은 예측 앞 차축이 GT 빨강일 때 정지선을 넘는 경우이고, 방향은 0.5 m 넘게 움직였을 때만 갱신한다(`signal_eval.crosses` 수정, 기존 신호 판독 394개 결과는 그대로).
  | 정지선 | 대기 n | 빨간불 위반 | 출발 n | 늦은 출발 | 출발 시점 중앙값 GT / 예측 |
  |---|---|---|---|---|---|
  | 7294 | 77 | 0 | 70 | 49 (70 %) | 1.1 s / 6.4 s 넘음 |
  | 6254 | 74 | 0 | 75 | 55 (73 %) | 1.1 s / 6.4 s 넘음 |
  | 4188 | 74 | 0 | 76 | 64 (84 %) | 1.2 s / 6.4 s 넘음 |
  | 4356 | 75 | 0 | 69 | 47 (68 %) | 1.0 s / 4.2 s |
  미확인 신호에서 넘은 경우는 0이다. 대기 중 예측이 움직인 것은 4356에서 1개이고 넘지는 않았다. 늦은 출발의 t0 라벨: 빨강 44/49, 48/55, 51/64(7294, 6254, 4188)이다. 신호가 t0 뒤에 초록으로 바뀌고 GT가 약 1 s 뒤 떠나는 경우라 입력 프레임에는 빨강만 있다. t0 라벨이 초록인데 정지한 것은 5, 7, 13, 9개다. 4356 늦은 출발 47개 중 35개는 t0 라벨이 노랑이다(좌회전 화살표 없음). GT가 화살표 없이 출발한 것이라 수집 운전기 쪽을 볼 일이다.
- **초록불 보고도 안 가나** (GPU 없이 저장된 예측만 사용, `logs/stageC/green_check/`: 판정 `judgments.csv`, 시트 `6254go_*.png`·`late_*.png`, 확대 `zoom4_L_*.png`, 연속 프레임 `seq_a/b.png`): 입력 마지막 프레임(Front, Cam7)을 직접 보고 판정.
  - 6254 출발 72개: 화면 빨강 30(예측 stay 20), 초록 5(stay 4), 안 보임 37(stay 28, 정지선 약 3 m 안이라 신호등이 Cam7 위로 벗어남). 그런데 72개 중 67개는 GT가 6.4 s 동안 0.5–1.1 m만 움직이고 정지선을 넘지 않는다. 출발이 아니라 빨간불 앞에서 조금씩 다가가는 동작이다. 실제 출발은 5개(한 장면, run1020k-w1-014 i974–990)뿐이고 화면은 초록, 예측은 4개가 stay다.
  - 6254 신호 기록: GetTrafficLightStatus 원본은 bag에 없다(`/collect_only/traffic_light_gt`는 live 채점기로만 흘러가고 녹화되지 않음). 라벨 빨강 16개는 경로 차선(측방 d 0.5–1.35 m)에 있던 3개 에피소드이고 화면도 전부 빨강이다. 미확인 56개는 7개 에피소드 모두 공사 회피로 옆 차선(d 3.1–4.1 m)에 있었다. 시뮬레이터는 차량이 올라 있는 링크의 신호만 보내므로, 6254에 매핑된 C1256W000091이 20–90 s 동안 한 번도 오지 않았다. 7개 중 6개는 에피소드가 정지선(경로 s 311–317 m, 정지선 315.9 m)에서 끝난다. 화면에서 보이는 경우는 빨강 14, 초록 5다.
  - 일반 늦은 출발 72개: 초록 11(stay 4), 빨강 12(stay 12), 노랑 7(stay 7), 안 보임 42(stay 35). 안 보임 중 26개는 에피소드 시작 직후(i ≤ 70)이고 78–90 m 앞 신호등이 너무 작다. 8개는 안개, 5개는 신호 없는 구간, 3개는 판정 불가다. 빨강 중 L40·L51·L61은 라벨이 t0에 초록으로 바뀌었지만 화면은 아직 빨강이다(화면이 약 0.2 s 늦음). 초록인데 stay인 4개는 안개 속 희미한 초록 2(L02, L11), t0 프레임에서 막 바뀐 1(L39, 앞 3프레임은 빨강), 맑은 날 초록 1(L65)이다.
- **t1c 뒤 정지**(사용자 "1-c 끝나면 멈춰", 17:20): 지금 도는 큐는 전체 단계라, `/workspace/stop_after_t1c.sh`가 `done/t1c`(B2 확인 뒤)를 보면 t2가 일을 시작하기 전에 큐를 멈춘다. 상자는 놀게 두고 vast 세션이 정지/종료를 정한다. 다음부터는 `echo <단계> > $OUT/stop_after`로 큐가 그 단계 뒤에 스스로 멈춘다.
- 학생 단계 연결(17:00에 해제, 지금은 쓰지 않음): 파이프라인이 `STAGES="t1t t1c t2 tgate"`로 돌 때 `/workspace/chain_student.sh`가 그 프로세스가 끝나고 `done/tgate`가 있고 HALT가 없으면 전체 단계로 다시 띄운다(끝난 단계는 건너뛰고 soft s1 sgate s2 qad deliver). 학생 파일은 10/25 11시 기준 워크스페이스와 md5가 모두 같다.
- 발산 정지는 원래대로 eval 손실 항목별이고, ~0인 값(eval_lm 1e-5 등)은 비율 오판을 막으려고 건너뛴다(최저값 ≤ 1e-3).

### B200 최적화와 단계 C 시간 (10/25, vast 1× B200, `scripts/profile_stage1.py`)

Stage 1-T 순전파+역전파, 실제 가중치, 같은 10 step 고정 배치(갱신 없음), GPU당 배치 4. 로그 B2 `runs/b200test/sw_*.json`.

| 변형 | s/step | s/샘플 | 최대 / 8 GPU 추정 메모리 | MFU | 같은 계산인가 | 판정 |
|---|---|---|---|---|---|---|
| A FA2, GC 켬 (기준) | 2.290 | 0.573 | 40.4 / 44.5 GB | 7.5 % | 기준 | |
| B FA2, GC 끔 | **1.082** | **0.271** | 111.4 / 115.6 GB | 15.9 % | 손실 10 step 동일, grad norm 차 ≤ 0.1 % | **채택** |
| C SDPA, GC 끔 | 1.351 | 0.338 | 114.7 / 118.8 GB | 12.7 % | 손실 차 ~0.1 %, grad norm 차 최대 9 % | 느림 |
| D SDPA cuDNN, GC 끔 | 1.472 | 0.368 | 114.7 / 118.8 GB | 11.7 % | **grad norm 전부 NaN** | 탈락 |
| E FA2, GC 끔, 배치 8 | OOM (178 GB) | | | | | 탈락 |

- 채택: Stage 1-T·1-C를 GPU당 배치 4 × 누적 1(recipe의 전체 배치 32 그대로), gradient checkpointing 끔, flash-attention 2 (`pipeline.sh`의 `T1_*` 기본값). 8 GPU ZeRO-2 추정 116 GB/GPU라 B200(192 GB)은 여유가 크고 H200(141 GB)도 들어간다.
- B에서 GPU 시간의 56 %가 GEMM, 37 %가 elementwise·norm·rotary(HF 구현의 융합 안 된 커널), GPU 점유 81 %. 더 줄일 여지(fused RMSNorm·rotary·SwiGLU·linear-CE 커널, torch.compile)는 재지 않았다.
- 1 GPU의 GPU당 ZeRO-2 몫 AdamW step은 9 ms. 데이터로더는 192 vCPU에서 워커 16개로 68 samples/s, 8 GPU 배치 4에 필요한 것은 약 30 samples/s.
- 다른 단계의 B200 ÷ H100 NVL: step 2 배치 1 1.25×, 배치 4 1.55×; Stage 2 배치 1 0.269 s (1.14×), 배치 2 sdpa 0.360 s (1.29×); soft target 배치 8 257 ms/샘플 (1.35×); 1-C 배치 1 GC 켬 0.520 s/샘플. 배치가 작으면 B200을 채우지 못한다.

**8 GPU 단계별 시간** (h; Stage 1은 위 B에 ZeRO-2 통신 5 %, Stage 2는 DDP 8 %, eval·저장 시간 포함. H200은 sm90 커널이라 H100 NVL 측정에서 추정한 범위)

| 단계 | 8× B200 | 8× H200 |
|---|---|---|
| t1t (2 epoch) | 5.9 | 8.6–10.9 |
| t1c (3 epoch) | 0.5 | 0.7–0.8 |
| t2 (3 epoch, GPU당 배치 1 = recipe) | 8.4 | 8.5–9.4 |
| tgate | 0.3–0.4 | 0.3–0.4 |
| soft (train 269k행, K=6) | 2.4–2.8 | 2.7–3.1 |
| 학생 s1·sgate·s2 (qad·deliver 미정) | 7–9 (FLOP 환산, 아래 "학생" 절의 시간) | 약 10–13 (H100 12–14 h에서) |
| 합계 | 약 33 | 약 40–44 |
| 여유 1.3배 | **약 43** | **약 52–57** |

- t2는 GPU당 배치 1이라 B200에서도 거의 빨라지지 않는다. 배치 2(sdpa, 전체 배치 16, recipe의 2배)로 돌리면 B200 약 5.7 h로 2.7 h 준다(채택하지 않음, 결정 필요).
- 학생 단계 시간은 학생 코드의 측정으로 바꾼다.

## 단계 B 결과 (2026-10-25, vast 57543085: 1× H100 NVL 94 GB, 300 GB 디스크)

로그·지표 사본: `logs/stageB/`(Mac), B2 `morai2026-alpamayo/runs/stageB/`(step 2 체크포인트 포함). 실행 스크립트: `scripts/stage_b.sh`(1–7), `scripts/stage_b_queue.sh`(후속).

| 항목 | 결과 |
|---|---|
| 1 테스트 | 23 passed |
| 2 release strict 로드 (BF16) | missing 0 (lidar 351), buffers 7, unexpected 0, mismatched 0, 14 s, 21.7 GB |
| 3 과적합 20샘플 300 step | step 2 traj CE 1.397 → 0.373 (0.547 s/step, 32.6 GB). Stage 2 FM 0.705 → 0.083 (0.309 s/step, 51.6 GB) |
| 4 배치·메모리 | step 2: 배치 1 0.528 s / 32.6 GB, 2 0.903 s / 32.7 GB, 4 1.587 s / 37.1 GB. Stage 2(fp32 master): 배치 1 0.306 s / 51.6 GB, 2(sdpa) 0.465 s / 51.7 GB |
| 5 IO | 8 workers 70.2 samples/s (토큰화까지) |
| 6 재개 | checkpoint-20에서 재개한 실행이 step 21/40부터 이어져 traj CE 1.296(10) → 1.261(20) → 1.282(30). checkpoint 하나가 71 GB(HF 22 + DeepSpeed model states 43 + optim 6.5)라, 데이터 189 GB와 같은 300 GB 디스크에서는 다시 돌릴 수 없었다 |
| 7 step 2 본 실행 | 269,052샘플(1,275 에피소드)에서 2000 step × 배치 4, 79분(1.61 s/step 단독). val traj CE(512샘플) 1.320(500) → 1.269(1000) → 1.241(1500) → 1.227(2000), 아직 내려가는 중. `final` 22 GB |
| Stage 1-T·1-C 스모크 (ZeRO-2 + CPU AdamW) | 9.374B 학습, 10 step 각 약 5.5 s/step, GPU 32.5 GB. 1-C는 CoC v1.1 키프레임(train 9,624, val 1,098)에서 LM 손실 2.16 → 1.22. 저장 → strict 재로드 통과, release 대비 최대 |Δ|: LLM·LiDAR·LM head 6.1e-5, 카메라 ViT 8e-6(lr 배율 0.1), expert 0 |
| Stage 1 순전파+역전파 (단독, 배치 1, GC) | 1-T 0.650 s/샘플(1,824토큰), 1-C 0.655 s/샘플(1,874토큰), 최대 39.3 GB (optimizer 제외) |
| soft target 생성 (BF16, CoC 없음, K=6) | Euler 10 step: 배치 1 438 ms, 배치 4·8 약 350 ms/샘플. 그중 프리필 약 145 ms, expert 1회(6 궤적) 약 20 ms. 고정 (x_t, t) 속도장 6점 = 프리필 + expert 1회 약 165 ms/샘플. 최대 41.2 GB |

### val 1,000샘플 (val + val_asset, 10번째 행마다 무작위, 6.4 s, BF16, CoC 없음, 6개 샘플)

| | ADE | FDE | minADE |
|---|---|---|---|
| 등속 (이력 마지막 0.5 s) | 5.14 | 12.77 | |
| 1.5 release, LiDAR = 카메라 ViT 사본 | 5.29 | 15.41 | 2.66 |
| 1.5 release, LiDAR 없음 | 5.29 | 15.45 | 2.67 |
| step 2, LiDAR 있음 | **9.50** | 23.74 | 7.55 |
| step 2, LiDAR 없음 | 5.29 | 15.45 | 2.67 |

셀별(ADE, release / step 2 LiDAR / 등속): C0 9.76 / 15.85 / 9.97 (68), C1 4.31 / 6.82 / 4.98 (36), C2 4.42 / 5.80 / 3.67 (208), C3 4.30 / 8.21 / 4.34 (479), C4 6.30 / 8.72 / 6.90 (37), C5 9.27 / 21.91 / 8.93 (69), C6 5.97 / 11.66 / 5.55 (103). 전체는 `logs/stageB/baseline_*/summary.json`.

teacher-forced 궤적 CE (같은 1,000샘플):

| | LiDAR 있음 | LiDAR 강제 제거 | 제거 − 있음 | 있음이 더 낮은 샘플 |
|---|---|---|---|---|
| 1.5 release (LiDAR = 카메라 ViT 사본) | 1.488 | 1.499 | 0.011 | 55.6 % |
| step 2 | **1.225** | 1.499 | **0.274** | **99.4 %** |

셀별 제거 − 있음(step 2): C0 0.45, C1 0.33, C2 0.23, C3 0.24, C4 0.37, C5 0.31, C6 0.29. 전체는 `logs/stageB/lidar_onoff*.json`.


- LiDAR는 이산 궤적 예측에 실제로 쓰인다: step 2 후 CE가 18 % 낮고(1.499 → 1.225) 거의 모든 샘플·모든 셀에서 낮다.
- 그런데 step 2는 LLM의 이산 궤적 토큰(CE)으로만 LiDAR 경로를 학습한다. 그래서 LiDAR 토큰이 바꾼 KV를, 카메라만 보던 KV로 학습된 고정 expert가 읽어 연속 궤적이 크게 나빠졌다. LiDAR를 빼면 release와 같다(학습한 것이 LiDAR 경로뿐이므로).
- 따라서 step 2 체크포인트의 expert 출력은 품질 지표로 쓰지 않는다(Stage 2가 expert를 새 KV에 맞춘다). step 2·Stage 1의 진척은 이산 CE로 본다.

### 단계 C 시간·비용 재추정 (8× H100 SXM, 10/24 시세 $19.53/h)

| 단계 | 근거 | 시간 |
|---|---|---|
| Stage 1-T | 269,052샘플/epoch × 0.65–0.72 s (측정 + ZeRO-2 통신 약 10%) ÷ 8 | 6.1–6.7 h/epoch |
| Stage 1-C | 9,624 × 3 epoch | 0.7 h |
| Stage 2 | 0.233 s(배치 2, sdpa)–0.306 s(배치 1) × 269,052 ÷ 8 | 2.2–2.9 h/epoch, 3 epoch 6.6–8.6 h |
| soft target | K=6 Euler 10: 0.35 s × 269,052 ÷ 8 / 속도장 6점만: 0.16 s | 3.3 h / 1.5 h |
| 학생 1·2단계 | FLOP 환산, 0.40–0.46 s/샘플 × 3 epoch (아래 "학생" 절) | 12–14 h |

- 합계(여유 없이) Stage 1-T 1 epoch 37–41 h, 2 epoch 43–48 h, 3 epoch 49–55 h. 학생 이전 추정 22 h 기준이다(10/25 학생 FLOP 환산 12–14 h로 바꾸면 각각 약 9 h 준다).
- 디버깅·관문 평가·재실행 여유 1.5배: **55–82 h, 약 $1,080–1,600**(이전 추정 60–85 h, $1,200–1,700과 같은 범위).
- Stage 1을 GPU당 배치 4 × 누적 1로 돌리면(step 2에서 샘플당 24% 빨랐다) 1-T가 epoch당 약 1.5 h 줄 수 있다. ZeRO-2로 그래디언트가 8등분되면 메모리는 들어갈 것으로 보지만 재지 않았다.

## 와우마요(WoWmayo, 학생) (MODEL_PLAN 4절 4–5단계): soft target, 학생 1·2단계

학생 모델의 이름은 **와우마요(WoWmayo)**다(사용자 결정, 10/25). 이 절의 "학생"은 와우마요를 말한다. 설정·경로·단계 이름(`student*`, s1/s2/qad/deliver)과 B2 경로는 그대로다.

Qwen3-VL-4B 학생(와우마요)을 T\*에서 증류하는 코드다. 학생은 `TeacherVLA`를 그대로 물려받는다(같은 processor·토큰, 이력 융합, LiDAR 라우팅, expert가 LLM KV를 읽는 경로, 추론). T\*는 같은 step 안에서 같은 배치(같은 토큰, 같은 LiDAR 마스크)로 고정된 채 돈다.

### 구성 (추가·변경)

| 파일 | 내용 |
|---|---|
| `morai_vla/student.py` | `StudentVLA`: 4B VLM + LiDAR ViT 사본 + T\*에서 이식한 expert·action_in/out_proj, 특징 증류 투영 2개(`vit_kd`). 손실 7개, `LinearLossWeight`(λ_v 일정) |
| `configs/student1.yaml`, `student2.yaml` | 학생 1단계(expert 고정), 2단계(expert 0.1배 학습률 + 속도장 KD, λ_v 1→0.2) |
| `configs/tiny/student.yaml` | `+tiny=student`: 같은 구조의 소형 무작위 학생(로컬 검증) |
| `scripts/soft_targets.py` | soft target 생성(BF16, CoC 없음, K=6 Euler 10 궤적 + 고정 (x_t, t) 6점의 속도장, seed 저장). torchrun으로 rank별 분할, 끊겨도 이어서 |
| `scripts/gate_student.py` | 관문: 같은 샘플·같은 초기 노이즈로 학생(이식 expert가 학생 KV를 읽음)과 T\*의 ADE, 비율, 학생–T\* 궤적 거리 |
| `scripts/parts_moved.py` | 실행의 시작 가중치를 저장된 config로 다시 만들어 `final`과 부위별 변화를 잰다 |
| `scripts/student_smoke_real.py` | 16 GB 카드용 실제 가중치 스모크(NF4 T\* 특징 저장 → CPU에서 학생 생성·이식 검사 → GPU 학생으로 손실 계산) |
| `scripts/student_mem.py` | 실제 폭·축소 깊이 모델로 학생 1 step(T\* 포함)의 메모리·시간 측정 → 전체 깊이 외삽 |
| `scripts/student_tiny.sh` | 로컬 검증 전체: 소형 T\* → 학생 1단계 → soft target → (임시) rule_ok → 2단계 → 관문 → 부위별 변화 |
| `scripts/pipeline_student.sh` | 단계 C 무인 큐(`pipeline.sh`)의 학생 단계 함수: `soft s1 sgate s2 qad deliver`와 각 `gate_*`, `metrics.json` |
| `scripts/student_dryrun.sh` | 위 함수를 GPU 1장에서 그대로 몇 step씩 돌리는 시험(`TINY=1` 소형, `TINY=0` 실제 크기 1× 스모크) |
| `morai_vla/rule_context.py`, `assets/rule_context.npz` | (Mac) 규칙 검사용 지도·장애물 묶음: 판정 실선 34,753개(도색 반폭), 판정 정지선 4개, 경로·속도 예외 구간, 바퀴·차체, 에피소드별 정적 장애물 2,109개 (0.9 MB) |
| `scripts/rule_check.py` | soft target 궤적 규칙 검사 → `rule_ok.parquet` (+ 규칙·셀별 탈락률 json). `--gt N`은 GT 미래로 보정 확인 |
| `morai_vla/qad.py`, `scripts/qad_search.py`, `scripts/qad_export.py`, `configs/student_qad.yaml` | QAD: ModelOpt AutoQuantize(1장) → 짧은 QAD 학습(`morai_vla.train`) → 관문·`quant_manifest.json`, `deliver` 묶음 |
| `tests/test_student.py`, `tests/test_rule_check.py` | 학생 11개(+ QAD 탐색·manifest·재로드, QAD 설정), 규칙 검사 3개(합성 지도에서 규칙마다 해당 궤적만 탈락) |
| `configs/student_ft.yaml`, `morai_vla/finetune.py`, `scripts/ft_*.py` | R2 미세조정(아래 "와우마요 R2 미세조정" 절) |
| `morai_vla/model.py` (교사, 동작 불변) | `load_config`·`exact_action_space`·`_velocity` 분리, `sample_trajectories(velocity_at=)`, fp32 재로드 때 LLM·expert가 bf16로 남던 것 수정 |
| `morai_vla/data.py` (교사, 기본 동작 불변) | `rule_ok`(parquet) → 샘플의 `vkd_ok` |

### 정한 것

**학생 만들기** (`StudentVLA.build`)
- 새 학생: Qwen3-VL-4B-Instruct 가중치를 읽고 토큰 행을 151,936 → 155,697로 늘린다(추가 3,761행은 HF 평균 초기화, LM head는 임베딩과 묶여 있다). LiDAR ViT는 4B 카메라 ViT(merger·DeepStack 3개 포함)의 사본이다.
- expert는 **4B text_config에 T\*의 `expert_cfg`를 덮어써서** 만들고, T\* 체크포인트의 `expert.*`, `action_in_proj.*`, `action_out_proj.*`를 텐서 하나씩 strict로 옮긴다. shape가 하나라도 다르거나, 빠진 파라미터·남는 키가 있으면 멈춘다. config에서 다시 만드는 버퍼(Fourier 주파수, rotary)는 T\*의 값과 같은지 붙일 때 검사한다. (a, κ) 통계는 같은 config.json에서 fp32로 만든다.
- 저장한 학생(2단계 시작, 평가)은 `init=<dir>`로 strict 로드한다. T\*는 학생 모듈에 등록하지 않은 속성이다. 그래서 저장·옵티마이저·DeepSpeed·dtype 변환에 끼지 않고, 첫 forward에서 같은 GPU로 옮겨진다.
- processor: 4B와 8B의 토크나이저·전처리 파일이 같다(chat template은 JSON 서식만 다름). 테스트가 같은 샘플의 텍스트·토큰·픽셀이 같음을 확인한다.

**손실** (`loss_weights` 키)

| 키 | 정의 | 1단계 | 2단계 |
|---|---|---|---|
| `kv` | 층마다 MSE(K) + MSE(V), 36층 평균(계획의 합 ÷ 36). expert가 읽는 프리픽스 [0, `<traj_future_start>`]의 실제 토큰만(왼쪽 패딩 제외). 투영 없음 | 1 | 1 |
| `vit` | 카메라·LiDAR ViT 패치 특징(main merger LayerNorm 출력, 2×2 병합 전)의 MSE(P(f_학생), f_T\*), P는 학습 전용 선형 1024 → 1152 | 10 | 10 |
| `fm` | GT flow matching. **expert를 거쳐 학생 KV로 그래디언트가 간다**(T\* Stage 2는 KV를 끊는다) | 1 | 1 |
| `vkd` | 같은 (x_t, t)에서 MSE(v_학생, v_T\*). `vkd_ok`(규칙 검사 통과) 행에만 | 0 | 1 → 0.2 |
| `traj_ce`, `lm` | 이산 궤적 CE, 그 밖의 답 토큰(CoC가 있으면 CoC, 끝 토큰) | 1 | 1 |
| `coc_kl` | CoC 답 토큰의 KL(T\* ‖ 학생) (선택) | 0 | 0 |

- **ViT 투영 방향을 계획(1152 → 1024)과 반대로 했다.** T\* 특징을 학생 공간으로 옮기는 학습 투영 P는 P와 학생 특징이 함께 0으로 가도 손실이 0이 된다(붕괴). 학생 → T\*(1024 → 1152)면 목표가 고정이라 그런 해가 없다(FitNets와 같은 방향). 파라미터 수는 같다.
- **속도장 KD의 목표는 같은 step의 T\*가 학생과 같은 (x_t, t)에서 바로 계산한다.** 계획 4절은 저장한 v를 쓰기로 했지만, 저장한 v는 지터 없는 명목 프레임에서 만든 것이라 학습 중 지터한 프레임과 입력이 다르다. T\*가 이미 같은 배치로 프리필을 하므로 추가 비용은 expert 1회(H100 약 20 ms)다. 저장한 v·seed는 오프라인 점검용으로 남긴다.
- **규칙 검사** (`scripts/rule_check.py`): 한 행은 T\* 궤적 K=6개가 모두 통과해야 `rule_ok`다. 게이트는 `data.train_dataset.rule_ok`(episode, i, rule_ok parquet)로 받고, 목록에 없는 행은 거짓이다. 2단계 학습에서 이 값이 없으면 멈춘다(평가에서는 vkd를 건너뛴다).
  - 규칙은 라벨을 만든 `tools/scorer` thresholds.yaml과 같다. (1) 속도: 0.1 s 구간 하나라도 60 km/h 초과(예외 구간 밖). (2) 차선: 바퀴가 판정 실선(도색 반폭 + 타이어 반폭)에 3 s 연속 닿음(GPS 음영 행 제외). (3) 신호: 앞축이 판정 정지선을 넘는 미래 행의 GT 신호가 적색이거나 모름. (4) 정적 장애물: 차체 상자와 시나리오 장애물 상자가 겹침.
  - 궤적은 t0 자차 좌표이고, t0 자세·경로 진행·GT 신호·음영은 그 에피소드 `_ego` 행에서 읽는다. 지도와 장애물은 `assets/rule_context.npz`(Mac에서 `morai_vla.rule_context`로 만듦)라서 학습 상자에는 따로 필요한 것이 없다.
  - 보지 않는 것: 보행자·NPC 차량(행별 궤적이 기록돼 있지 않음), 주행 불능. 장애물 크기는 근사다: 소품 1.0 m × scale, 주차 차량 4.6 × 1.9 m(에셋 크기 표가 없다, DATA_PLAN 10/21).
  - 보정: train 행 3,000개의 GT 미래(모두 traj_ok)를 넣으면 99.73 % 통과(차선 0.20 %, 장애물 0.07 %, 속도·신호 0)라 거짓 탈락이 적다. 8× 상자에서 269k행 검사는 수 분이다(KD 트리, 행렬 연산).
- **QAD** (`morai_vla/qad.py`, MODEL_PLAN 4절 6단계, 5.1절): NVIDIA ModelOpt(`nvidia-modelopt==0.47.0`, PyPI)로 순전파만 배포 정밀도를 흉내 낸다(STE, 가중치·역전파는 BF16/fp32).
  - LLM 디코더·expert 선형층: 층마다 NVFP4(16값 블록, e4m3 블록 스케일) 또는 FP8(텐서당)을 AutoQuantize가 고른다. 민감도는 학생 자신의 학습 손실(T\* 증류 + GT)의 그래디언트다. BF16은 고르지 못하게 했다. 예산은 이 층들만으로 센 `quant.effective_bits` = 4.75다(10/26 사용자 결정: 가중치 약 7 %만 FP8, 나머지 NVFP4, 5080 벤치마크 배치; 처음 값은 recipe 예시 6.5). 아래 스모크·소형 수치는 6.5 때 것이다. ModelOpt가 q/k/v와 gate/up을 같은 형식으로 묶는다.
  - 카메라 ViT 블록은 FP8로 고정한다. LiDAR ViT, merger·DeepStack, 임베딩·LM head, action in/out proj, KD 투영, KV 캐시는 BF16이다. Euler 적분은 모델 밖에서 FP32다.
  - 탐색은 한 프로세스(GPU 1장)에서 돈다. 모든 rank가 같은 형식을 써야 하기 때문이다. 결과는 학생 HF 폴더와 `modelopt_state.pth`(다시 읽으면 양자화기가 복원됨)로 저장하고, 8장 QAD 학습이 그것에서 시작한다. QAD 학습은 2단계 손실 그대로(λ_v 0.2 고정), lr 1e-6, 800 step(약 0.1 epoch)이다.
  - 내보내기(`deliver`): QAD 학생 폴더(BF16 가중치 + 양자화기 amax), `modelopt_state.pth`, `quant_manifest.json`(층별 형식, 블록 크기, weight·input amax, shape; NVFP4 텐서 스케일 = amax / (6·448), FP8 = amax / 448), `summary.json`. 5080 엔진 변환은 이 manifest의 형식과 amax를 쓴다(10/24 엔진의 활성값 스케일 자리 표시값 1.0을 대신한다).
  - FP8 가짜 양자화의 CUDA 확장은 nvcc가 없으면 빌드되지 않지만, 그때도 `torch.float8_e4m3fn` 변환으로 실제로 양자화한다(느릴 뿐). NVFP4는 Triton 커널(sm89 이상)이다.
- **CoC**: 기본 설정은 CoC 없음(추론 형식)이다. `data.train_dataset.coc=true`로 짧게 돌리면 CoC CE와 선택 KL이 걸린다. 프리픽스에 CoC가 든 행은 `fm`·`vkd`를 건너뛴다(expert는 CoC 없는 프리픽스만 안다).
- **GT FM의 `traj_ok`**: 인덱스에 traj_ok 행만 있으므로 따로 거를 것이 없다.
- **gradient checkpointing 끔**: expert가 학생 KV 캐시를 그래디언트째 읽어야 하는데, HF는 checkpointing 중에 캐시를 버린다. 대신 메모리는 아래 계획처럼 배치 1로 맞춘다. flash-attention 2는 배치 1에서 그대로 쓴다(왼쪽 패딩이 있는 배치는 sdpa).
- **로짓**: 학생 로짓은 첫 답 토큰부터만 계산한다(1,856 위치 × 155,697 대신 약 131 위치). 샘플당 약 2 GB를 아낀다.
- **손실 가중치 (출발값)**: 실제 가중치 스모크의 시작 크기(아래)로 정했다. KV 층 평균 약 13.5, 궤적 CE 약 22, FM 5–16, ViT ×10 약 1.
- **학습률·epoch (출발값)**: recipe Stage 1 값(1e-5, warmup 500, cosine, GPU당 1 × 누적 4, 카메라 ViT 0.1배). 1단계 2 epoch, 2단계 1 epoch을 다 돈다(조기 종료 없음). 2단계 expert·action proj는 0.1배.
- **1단계 `final/` = 궤적 ADE 최저 측정** (사용자 결정, 10/25 저녁, MODEL_PLAN 3.4): 교사 Stage 2와 같은 `morai_vla/early_stop.py` `TrajEarlyStop`을 쓴다(`configs/student1.yaml`).
  - 600 step(H200에서 약 20분, GPU 수와 무관)마다 고정 val 1,000샘플에서 학생 KV 위 고정 expert의 6.4 s 궤적 ADE(K=6, 학생 관문과 같은 지표)를 잰다. 2 epoch이 상한이다.
  - 0.5 epoch 뒤로 그때까지 최고보다 2 % 넘게 낮아지지 않은 측정이 3번 연속이면 멈춘다. 그 뒤 500 step(8장 기준 2 epoch 약 16.8k step의 3 %, Stage 2와 같은 비율) 동안 LR을 0까지 선형으로 낮춘다.
  - 최고 측정과 감쇠 후 중 ADE가 낮은 쪽이 `final/`이고, 기록은 `ade_select.json`·`ade_early_stop.json`(그리고 `metrics.json` `final`)이다.
  - 측정마다 학습 파라미터 스냅샷 9.7 GB를 남긴다(`ade_snapshots/`, 최대 약 32개 310 GB; pipeline이 B2 push 뒤 지운다).
  - 학생 쪽 측정 코드는 따로 없다. `StudentVLA`가 `sample_trajectories`를 물려받으므로 공통 `expert_ade`가 그대로 학생 KV 위 고정 expert 궤적을 잰다. 공통 설정은 `configs/student_base.yaml`로 옮겼고, 1단계만 이 조기 종료를 더한다.
- **2단계·QAD `final/` = 최고 체크포인트** (사용자 결정, 10/25): eval GT flow matching 손실(`eval_fm`, expert를 거친 값; QAD는 가짜 양자화 상태)이 가장 낮은 체크포인트다. 가중합 eval_loss가 아닌 이유는 2단계 λ_v 일정 때문에 총합이 품질과 상관없이 내려가기 때문이다.
  - `EvalComponentMetric`이 모든 rank의 평균 `eval_fm`을 평가 지표에 넣는다. 그래야 rank가 같은 체크포인트를 고른다.
  - 평가의 노이즈와 t는 샘플 키로 고정한다(학습 RNG는 그대로). 그래서 체크포인트끼리 같은 추첨으로 비교된다.
  - 저장한 평가만 고를 수 있으므로 eval_steps = save_steps다(2단계 1000, QAD 200; 1단계는 2000으로 재개용). 고른 step과 값은 단계의 `metrics.json` `final`에 남는다.

### 로컬 검증 (winpc RTX 5080, 2026-10-25)

환경: `~/claude-bench/venv-teacher`(교사와 같음), 코드 사본 `~/morai-student/train`, 결과 `~/morai-student/runs/`(손실·관문·부위 변화·스모크·메모리 json 사본은 `logs/student/`). 소형 실행은 다른 세션의 NF4 평가와 GPU를 나눠 써서 step 시간(1–5 s)은 의미가 없다.

**테스트**: `python -m pytest -q tests` → 42 passed(10/25 저녁, 그중 학생 9개). 학생 테스트가 확인하는 것:
- 4B processor가 T\*(8B)와 같은 텍스트·토큰 id·픽셀을 낸다.
- 소형 T\*를 저장하고 학생을 만들면, 학생 expert가 **T\*의 KV를 받았을 때 T\* expert와 비트 단위로 같은 속도장**을 낸다. shape가 다른 expert에서 이식하면 `size mismatch`로 멈춘다.
- 1단계: 손실 5개가 모두 유한·양수이고, 그래디언트는 카메라 ViT·LiDAR·LLM·KD 투영에만 가고 expert와 T\*에는 없다. **FM만 켜도 LLM·카메라 ViT에 그래디언트가 간다**(KV를 끊지 않음).
- 2단계: `vkd_ok` 없이 학습하면 멈추고, 있으면 통과한 행에만 vkd가 걸리고 expert에 그래디언트가 간다. 모두 거짓이면 vkd = 0. CoC 행은 FM을 건너뛰고 `coc_kl` > 0.
- 저장 → `init=<dir>` strict 재로드가 모든 텐서를 그대로 돌려주고, LM head 묶음이 유지되며, T\*는 state_dict에 없다.
- 설정 4개(student1, student2, 각각 `+tiny=student`)가 구성되고 키가 맞는다.

**소형 무작위 모델로 20샘플 과적합** (`scripts/student_tiny.sh`, 같은 20 train 샘플, CoC 없음, LiDAR dropout 0.15, 배치 2)
- 소형 T\*: 교사 `tiny_overfit`(8B 설정 축소, 텍스트 256 · ViT 128 · expert 128, 4층 · KV 2×64)을 CoC 없이 1,000 step. traj CE 11.26 → 0.033, FM 2.80 → 2.48(작은 무작위 expert가 노이즈를 읽지 못해 FM이 내려가지 않는 것은 위 "로컬 검증"과 같다).
- 학생: 4B 설정 축소(텍스트 192 · ViT 96, 4층 · KV 2×64, 같은 expert). 투영은 96 → 128이다. T\* expert 60텐서를 strict로 옮겼다.

최종 손실 설정(kv 층 평균, vit ×10)으로 돌린 값이다(`runs/tiny2/`). GT FM을 끈 줄만 그 전 설정(kv 4층 합, vit ×1)이다.

| | step | kv | vit | traj_ce | lm | fm | vkd |
|---|---|---|---|---|---|---|---|
| 1단계 | 10 | 2.287 | 2.133 | 11.450 | 9.794 | 2.682 | — |
| 1단계 | 1000 | **0.070** | **0.0044** | **0.030** | 0.0003 | 1.865 | — |
| 2단계 | 10 | 0.069 | 0.0044 | 0.031 | 0.0003 | 1.724 | 0.226 |
| 2단계 | 1000 | **0.041** | **0.0026** | **0.0015** | 0.00001 | 1.579 | 0.360 |
| 2단계, GT FM 끔 (400 step, 이전 설정) | 250 | 0.063 | 0.0052 | 0.0039 | 0.00001 | 2.481 | **0.0004** |

- KV·ViT·CE 손실은 0 근처로 간다. 1단계 끝 kv는 처음의 3 %, vit는 0.2 %이고, 2단계에서도 계속 내려간다. 처음 보는 val 8샘플(eval)도 kv 0.061, vit 0.0039로 같이 내려가서 외운 것만은 아니다.
- FM은 0으로 가지 않는다. 1단계 expert가 고정된 소형 T\* expert라서 T\* 자신의 FM(2.48) 근처에 머문다.
- vkd는 GT FM(가중치 1)과 당긴다. T\*의 v가 GT 목표와 다르므로, λ_v가 0.2로 줄수록 학생은 GT 쪽으로 가고 vkd가 0.36까지 오른다. 이것이 설계("주 손실은 GT")다. GT FM을 끄면 vkd가 0.20 → 0.0004로 내려가므로 경로 자체는 맞다.
- 부위별 최대 |Δ| (`scripts/parts_moved.py`, 시작 가중치를 저장된 config로 다시 만들어 비교)

| | 카메라 ViT | LiDAR ViT | LLM(임베딩 포함) | KD 투영 | expert·action proj | (a, κ) 통계 |
|---|---|---|---|---|---|---|
| 1단계 | 0.0274 | 0.212 | 0.294 | 0.147 | **0** | 0 |
| 2단계 | 0.0212 | 0.190 | 0.101 | 0.094 | 0.0125 (평균 0.0013, LLM 평균 0.0123의 약 1/9) | 0 |

- 관문(`scripts/gate_student.py +dataset=train_dataset`, 같은 20샘플, K=6, 같은 초기 노이즈): T\* ADE 4.55 m. 학생 시작 4.54 m(비율 1.00, st_ade 1.96), 1단계 뒤 2.45 m(0.54, st_ade 3.15), 2단계 뒤 2.44 m(0.536, st_ade 3.10).
  - 학생이 이 소형 T\*보다 낫다. 1단계 GT FM이 고정 expert를 거쳐 학생 KV를 GT 쪽으로 당기기 때문이다. T\*가 약하면 관문은 "비율 ≤ 1"로 통과하지만 st_ade는 커진다. 실제 T\*에서는 비율과 st_ade를 함께 본다.
- soft target: 20샘플 모두 생성(소형 T\*, 배치 4, 363 ms/샘플). 샘플마다 궤적 6 × 64 × (x, y, yaw), t ∈ (0.009, 0.993), 속도장 6 × 64 × 2, seed. `rule_ok`는 검사기가 없어 시험용 해시 규칙(70%)으로 만들었다(18/20 통과).

**실제 가중치 스모크** (`scripts/student_smoke_real.py`, 1.5 release를 T\* 대역으로, val 4샘플, `runs/real_smoke/smoke_real.json`)
- T\*: NF4(LLM·expert 선형층), 9.4 GB, 로드 143 s. 프리픽스 1,721–1,728토큰의 36층 KV, ViT 특징, 고정 (x_t, t) 6점의 속도장, 궤적 2개를 저장했다.
- **4B + 이식 expert의 strict 로드 (MODEL_PLAN 7절의 미확인 항목)**: CPU에서 135 s. release의 expert·action proj 409텐서(expert 397, in 10, out 2)를 strict로 옮겼고, 다시 읽어 비교한 불일치 0. 4B text_config로 만든 expert 설정과 8B로 만든 설정의 차이는 `tie_word_embeddings`, `vocab_size` 두 개뿐이다(expert에는 임베딩이 없다). config로 만드는 버퍼는 T\*와 모두 같다. 토큰 행 151,936 → 155,697, LM head 묶음 유지. 전체 7.146B(카메라 ViT 0.415, LiDAR 0.415, LLM 4.032(임베딩 0.399 포함), expert 2.281, KD 투영 0.002).
- 학생 BF16을 GPU에 올리면 13.4 GiB이고, 손실 계산과 궤적 추론까지 최대 14.9 GiB다(16 GB 카드에서는 추론만 된다).
- **시작 시점 손실 크기** (4샘플 평균; 학생은 학습 전 4B + 이식 expert)

  | 항목 | 값 |
  |---|---|
  | KV MSE 층 평균 (K + V) | 13.5 (합 486). 0층 K 하나가 201로 합의 41 %. V는 앞 층 0–0.1, 뒤 4층 15–20 |
  | 같은 층 T\* K, V 제곱 평균 | 1.1–14 (층 합 89). 학생과의 MSE가 그 3–6배다. 두 모델의 K, V가 서로 다른 기저에 있어서 0을 예측하는 것보다도 멀다 |
  | 카메라 ViT 특징: T\* 제곱 평균 / 시작 학생에 최적 선형(1024 → 1152) 잔차 / R² | 0.071–0.076 / 0.024–0.034 / 0.53–0.66 |
  | 답 토큰 CE (학생 / T\*) | 22.0 / 0.3–2.9 (궤적 토큰 행이 새것) |
  | GT FM (학생 KV / T\* KV, 같은 expert, 같은 (x_t, t)) | 5.5–16.3 / 1.6–8.3 |
  | vkd (학생 v − T\* v) | 5.6–10.6 |

  - LiDAR 특징의 선형 적합은 패치 400개 < 차원 1,025라 의미가 없어서 뺐다.
  - 이 크기를 보고 손실 가중치 출발값을 정했다. KV는 계획의 층 합 대신 **층 평균**으로 해서(합의 1/36) CE·FM과 같은 자릿수에서 시작한다. ViT KD는 목표가 LayerNorm 출력이라 작으므로 **λ_vit = 10**으로 해서 약 1에서 시작한다. 0층 K가 오래 지배하면 층별 정규화를 ablation으로 본다.

**무인 큐 함수 시험** (`scripts/student_dryrun.sh`, 5080, 소형 모델, 단계마다 40–200 step, `runs/dry/`): `pipeline_student.sh`의 실제 함수 `soft s1 sgate s2 qad deliver`가 끝까지 돈다.
- soft 20샘플(352 ms/샘플). s1·s2는 `metrics.json`을 쓴다. sgate 비율 1.05(val 8샘플). s2는 시작할 때 규칙 검사를 돌린다.
- qad: AutoQuantize 6.48 bits(NVFP4 21층, FP8 51층, 카메라 ViT 포함) → QAD 40 step → 관문 → manifest.
  - 이 소형 모델에서는 QAD 학생 ADE가 2단계 학생의 1.45배라 `gate_qad`가 의도대로 멈췄다. 폭 96–256의 층은 NVFP4 오차가 실제 크기(2,560)보다 훨씬 크고, 8샘플 40 step이다. 실제 크기 판정은 1× 스모크와 본 실행에서 본다.
- deliver 묶음: HF 폴더, `modelopt_state.pth`, `quant_manifest.json`, `summary.json`, `CONTENTS.txt`.

**1단계 궤적 조기 종료 시험** (10/25 저녁, 5080, 소형 모델 40 step, 측정 10 step마다 val 16샘플, 감쇠 5 step, `scripts/student_dryrun.sh`, 로그 `logs/student/early_stop/`)
- 멈춤을 강제한 실행(최소 0 epoch, 연속 1번, 2 % 대신 100 %): step 20에서 멈춤 → LR 0.994배에서 step 25까지 0으로 → 감쇠 후 측정(5.98 m)이 최저 → final/. 이어서 sgate 통과(비율 0.92).
- 사용자 규칙 그대로(0.5 epoch = 5 step, 2 %, 3번): ADE 6.09 → 6.30 → 6.23 → 6.62로 20·30·40에서 연속 3번 나아지지 않아 상한(step 40)에서 멈췄고, final/은 앞선 step 10 스냅샷이다.
- 두 실행 모두 final/의 학습 파라미터 200개가 고른 스냅샷과 정확히 같다(최대 차이 0). `metrics.json` `final`에 고른 step·ADE·멈춘 step이 남는다. 테스트 58 passed.

**실제 크기 1× 스모크** (vast 57587913, 1× H100 NVL 94 GB, 10/25, `TINY=0 scripts/student_dryrun.sh`, 로그 `logs/student/h100_smoke/`)
- 조건: 실제 4B 학생 + 이식 expert, T\* 대역은 1.5 release(BF16). 단계마다 30 step이고, GPU 1장이라 SGD로 DeepSpeed 없이 돌렸다. 데이터는 27개 에피소드 부분집합이다.

| 단계 | 결과 | s/샘플 | 최대 GiB |
|---|---|---|---|
| soft | 64샘플(배치 8) | 0.445 | — |
| s1 | 통과. 최고 체크포인트 선택 동작 | **0.593** | 62.1 (계획 62.9) |
| sgate | 비율 1.57로 정지. 30 step의 학생(5.45 m)과 T\*(3.47 m)의 차이라 의도대로다 | — | — |
| s2 | 통과. release 궤적의 규칙 검사 통과율 77 %(속도 9, 신호 12, 차선 3, 장애물 2 % 탈락) | **0.645** | 66.6 |
| qad | AutoQuantize 77 s, 6.50 bits(NVFP4 240층 + FP8 360층). QAD 학습 후 `ratio_vs_s2` 0.99 | 0.90 | 79.9 |
| deliver | 14.3 GB | — | — |

- DeepSpeed ZeRO-2(CPU offload, 1장): s1 10 step(54 GiB), QAD(양자화기 복원) 10 step(67 GiB) 모두 오류 없다.
- 배치 2(sdpa)는 0.56 s/샘플, 83 GiB로 배치 1(FA2)의 1.07배뿐이라 쓰지 않는다.
- 상자에서 겪은 것: HF 캐시에 `refs/main`이 없어 오프라인 로드가 실패했다(손으로 넣음). 150 GB 디스크가 스모크 체크포인트로 찼다.
- **시간 다시 계산 (7× H200, H200 ≈ H100 NVL × 1.15, ZeRO-2 통신 5 %)**: soft 약 3.2 h, s1 2 epoch 11.6–12.6 h, sgate 0.25 h, s2 6–7 h, qad 약 1.3 h로 **합 22–24 h**다. FLOP 환산(아래)보다 약 30 % 느리다.

### 메모리 계획 (8× H100 80 GB, GPU당)

T\* 11.655B, 학생 7.146B(1단계 학습 4.864B, 2단계 7.146B, LM head 묶음), 샘플 1,852토큰(이미지 1,344 + BEV 100 토큰, 패치 5,376 + 200), GPU당 배치 1, gradient checkpointing 없음. 단위 GiB.

**활성값 실측** (`scripts/student_mem.py`, 5080, 실제 폭 · 줄인 깊이 · 실제 MORAI 샘플, 무작위 가중치, `runs/mem2.json`). 순전파가 끝난 시점의 상주 가중치 위 메모리(활성값 + T\* KV·ViT 특징). ZeRO-2는 그래디언트를 버킷으로 보내므로 이것에 버킷만 더하면 된다.

| (LLM·expert 층 L, ViT 층 D) | (2, 3) | (5, 3) | (2, 9) | 층당 | L=36, D=24 외삽 |
|---|---|---|---|---|---|
| 1단계 | 1.87 | 2.94 | 3.24 | LLM 0.357, ViT 0.228, 나머지 0.47 | **18.8** |
| 2단계 | 1.80 | 2.91 | 3.19 | LLM 0.370, ViT 0.232, 나머지 0.36 | **19.2** |

**GPU당 합계 (8× H100 80 GB)**

| 항목 | 계산 | 1단계 | 2단계 |
|---|---|---|---|
| 학생 BF16 가중치 (GPU마다 전체) | 7.146B × 2 B | 13.3 | 13.3 |
| ZeRO-2 상태: fp32 원본 + Adam m·v + fp32 그래디언트, 8등분 | 16 B × 학습 파라미터 ÷ 8 | 9.1 | 13.3 |
| T\* BF16 가중치 (GPU마다 전체, 고정) | 11.655B × 2 B (단계 B 실측 21.7) | 21.7 | 21.7 |
| 학생 활성값 + T\* KV·특징 (위 외삽) | | 18.8 | 19.2 |
| 역전파 초기 묶인 임베딩 그래디언트 + reduce 버킷 2×2e8 bf16 | 0.74 + 0.75 | 1.5 | 1.5 |
| CUDA context, 할당기 단편화 | 추정 | 2–3 | 2–3 |
| **합** | | **66–67** | **71–72** |

- **고른 것**: T\*를 GPU마다 통째로 둔다(샤딩 없음). 학생은 ZeRO-2(교사 Stage 1과 같은 `zero2.json`), GPU당 배치 1 × 누적 4, flash-attention 2, checkpointing 없음. 80 GB에 1단계 약 13 GiB, 2단계 약 8 GiB 여유가 남는다. T\*를 ZeRO-3으로 나누면 약 19 GiB를 아끼지만 step마다 22 GiB를 all-gather해야 하므로, 필요해질 때만 쓴다.
- **배치 2는 80 GB에 안 들어간다**(+19 GiB → 85–91). H200(141 GB)·B200(192 GB)에서는 배치 4까지 들어가지만(2단계 약 129 GiB), 배치 2 이상은 왼쪽 패딩 때문에 `model.attn_implementation=sdpa`가 필요하다(expert 패딩 마스크는 flash-attention 2에서 안 됨). B200 교사 측정에서 배치 1과 4의 샘플당 시간이 비슷했으므로(1-C 배치 1 0.520 s, 1-T 배치 4 GC 0.573 s) 배치 1 + FA2를 기본으로 둔다.
- **넘칠 때 줄이는 순서** (어느 것도 아직 구현하지 않았다): (1) `coc_kl`이 0이면 T\* LM head가 필요 없다(−1.2 GiB). (2) 학생 ViT 두 개에만 checkpointing을 건다(ViT 활성값 5.5 → 약 0.5 GiB). KV 캐시는 LLM 쪽이라 영향이 없다. (3) T\* 샤딩(−19 GiB). (4) 옵티마이저 CPU 오프로드(`zero2_offload.json`, 느림).
- 5080 실측이 맞는지는 런북 4(8장 30 step)의 `peak_mem_gb`로 확인한다.

**시간 (FLOP으로 환산, 측정한 교사 수치 기준)**: 5080의 줄인 깊이 실행은 층 수를 바꿔도 T\* 순전파 0.15–0.19 s, 학생 순전파+역전파 0.30–0.38 s로 거의 같았다. 고정 오버헤드가 대부분이라 외삽에 쓰지 않았다.
- 샘플당 연산: T\* 순전파 약 30 TFLOP(LLM 25.7, ViT 4.6). 학생 순전파+역전파 약 51 TFLOP(LLM 13.4, ViT 3.4, expert 0.3, ×3). 2단계 vkd는 T\* expert 1회(0.3)를 더한다. 합 약 81 TFLOP다. 교사 Stage 1 한 샘플(약 98 TFLOP)의 0.83배다.
- **8× B200**: 교사 1-T(GC 끔, 배치 4)가 0.271 s/샘플(362 TFLOPS)이었으므로 학생은 0.22–0.29 s/샘플이다(두 모델·배치 1 오버헤드 ×1.0–1.3). epoch(269,052행)당 2.2–2.9 h(ZeRO-2 통신 5 % 포함)다. 1단계 2 epoch 4.4–5.7 h, 2단계 1 epoch 2.2–2.9 h, 관문 2회 약 0.5 h로 **합 7–9 h**다. 파이프라인 표의 "약 15 h" 자리 표시값을 대신한다.
- **8× H100**: T\* 순전파는 soft target 측정 프리필 145 ms(배치 4–8, 배치 1이면 약 0.2 s), 학생 51 TFLOP은 교사 1-T의 약 200 TFLOPS로 0.25 s다. 그래서 0.40–0.46 s/샘플, epoch당 3.9–4.5 h, **3 epoch 12–14 h**다(이전 추정 22 h).
### 단계 C 학생 런북

T\*가 3.5 관문을 통과한 뒤 돈다. 무인 실행에서는 `scripts/pipeline.sh`의 `soft`·`s1`·`sgate`·`s2` 단계(`scripts/pipeline_student.sh`)가 아래 명령을 그대로 부른다(위 "단계 C 파이프라인"). T\*가 없을 때 개발용으로는 `TSTAR`에 1.5 릴리스를 쓴다(step 2 체크포인트는 LiDAR가 있으면 expert 출력이 나빠서 관문 숫자가 의미 없다, 위 "단계 B 결과").

**0. 준비**: 단계 B 런북 0과 같은 셸(`$C`, `$OUT`)에 `export TSTAR=<T* 최종 체크포인트(Stage 2 결과)>`. 학생 VLM `Qwen/Qwen3-VL-4B-Instruct`(공개, 8.3 GB)를 GPU를 빌리기 전에 `$HF_HOME`에 받아 둔다.

**1. 테스트**: `python -m pytest -q tests` → 42 passed(10/25). 4B·8B processor와 릴리스 `config.json`이 HF 캐시에 있어야 한다(없으면 `ALPAMAYO15=<릴리스 dir>`).

**2. soft target**
```bash
torchrun --nproc_per_node 8 scripts/soft_targets.py --config-name student1 $C paths.teacher=$TSTAR +out=$OUT/soft +batch=8
```
- 학생 1단계와 같은 train 행(269,052)이다. H100에서 샘플당 약 0.35 s라 8장이면 약 3.3 h다(단계 B 측정). 끊기면 같은 명령으로 이어서 돈다(완성된 part는 건너뛴다).
- 결과 `part-RR-NNNNN.npz`: `key, cell, lidar_dropped, traj_seed, seed, t (6), v (6×64×2), xyyaw (6×64×3)`.

**3. 규칙 검사** (2단계 직전)
```bash
python scripts/rule_check.py --parts $OUT/soft --root /data/step0 --out $OUT/rule_ok.parquet   # + rule_ok.parquet.json: 규칙·셀별 탈락률
```
- 1단계는 이것 없이 돈다. 무인 큐에서는 `s2` 함수가 시작할 때 돌린다.

**4. 짧은 실측** (본 실행 전, 30 step)
```bash
P="trainer.max_steps=30 trainer.logging_steps=10 trainer.eval_strategy=no trainer.save_strategy=no trainer.load_best_model_at_end=false save_final=false"
torchrun --nproc_per_node 8 -m morai_vla.train --config-name student1 $C paths.teacher=$TSTAR paths.output_dir=$OUT/prof_s1 $P
```
- `loss_components.jsonl`의 `peak_mem_gb`가 위 메모리 계획과 맞는지, `s_per_step`이 시간 추정과 맞는지 본다. 넘치면 위 "넘칠 때" 순서로 줄인다.

**5. 학생 1단계**
```bash
torchrun --nproc_per_node 8 -m morai_vla.train --config-name student1 $C paths.teacher=$TSTAR paths.output_dir=$OUT/student1
```
- 손실 `kv, vit, fm, traj_ce, lm`과 500 step마다 `eval_*`(val+val_asset 512)를 본다. `kv`가 주 지표다.

**6. 관문** (MODEL_PLAN 4.5)
```bash
G="python scripts/gate_student.py --config-name student1 $C paths.teacher=$TSTAR +n=1000"
$G +student=start +out=$OUT/gate0                     # 1단계 전 (4B + 이식 expert)
$G +student=$OUT/student1/final +out=$OUT/gate1
python scripts/parts_moved.py $OUT/student1             # expert 0이어야 한다
```
- `summary.json`의 `ratio`(학생 ADE / T\* ADE)와 `st_ade`(같은 노이즈의 학생–T\* 궤적 거리)를 본다. 출발 기준: ratio ≤ 1.1, 셀별로도 크게 벗어나는 셀이 없을 것. 못 미치면 MODEL_PLAN 4.5의 조건부 PAI 사전 증류를 검토한다.

**7. 학생 2단계**
```bash
torchrun --nproc_per_node 8 -m morai_vla.train --config-name student2 $C paths.teacher=$TSTAR model.init=$OUT/student1/final \
    data.train_dataset.rule_ok=$OUT/rule_ok.parquet paths.output_dir=$OUT/student2
$G +student=$OUT/student2/final +out=$OUT/gate2; python scripts/parts_moved.py $OUT/student2
```
- `vkd`(규칙 통과 행만)와 GT `fm`을 함께 본다. λ_v는 `LinearLossWeight`가 step마다 1 → 0.2로 바꾼다.

**8. QAD** (ModelOpt 필요: `pip install --no-deps nvidia-modelopt==0.47.0 "pulp<4" nvidia-ml-py`)
```bash
Q="--config-name student_qad $C paths.teacher=$TSTAR data.train_dataset.rule_ok=$OUT/rule_ok.parquet"
python scripts/qad_search.py $Q model.init=$OUT/student2/final +out=$OUT/qad/search            # GPU 1장, AutoQuantize
torchrun --nproc_per_node 8 -m morai_vla.train $Q model.init=$OUT/qad/search paths.output_dir=$OUT/qad
$G +student=$OUT/student2/final +out=$OUT/qad/gate_s2 +n=500; $G +student=$OUT/qad/final +out=$OUT/qad/gate_qad +n=500
python scripts/qad_export.py summary $OUT/qad && python scripts/qad_export.py deliver $OUT/qad $OUT/deliver
```
- `summary.json`의 `ratio_vs_s2`(QAD 학생 ADE / 2단계 학생 ADE, 같은 샘플·노이즈)가 1.25 미만이어야 한다. `quant_manifest.json`의 `effective_bits_searched`가 예산 근처인지 본다.

**무인 큐**: 위 2–8은 `scripts/pipeline_student.sh`의 `soft s1 sgate s2 qad deliver`가 같은 명령으로 돈다(정지 기준: 학습 손실 비유한값, `eval_kv`·`eval_traj_ce` 최저의 1.2배 초과, sgate `ratio` 1.5 이상, qad `ratio_vs_s2` 1.25 이상). 같은 함수를 GPU 1장에서 몇 step씩 시험하려면 `scripts/student_dryrun.sh`(소형 `TINY=1`, 실제 크기 `TINY=0`)를 쓴다.

## 와우마요 R2 미세조정 (DATA_PLAN_R2.md 7절, 2026-10-27)

레시피는 DATA_PLAN_R2.md 7절 '학습' 표다(사용자 10/27 "제안대로 가자"). 아래 표의 값은 사용자 확인 없이 바꾸지 않는다. s2 final(BF16, QAD 전)에서 시작하는 1단계 공동 학습이다. 그 뒤 순서는 QAD(800 step) → 바로 납품 → 관문이다(사용자 10/27 규칙, 10/28 "납품이랑 관문 순서가 반대야").

### 구성

| 파일 | 내용 |
|---|---|
| `configs/student_ft.yaml` (+ `configs/ft_reset.yaml`) | 미세조정 설정. `ft_reset`은 base.yaml의 단일 train 데이터셋 키를 지우고 혼합으로 바꾸기 위한 빈 노드다 |
| `morai_vla/finetune.py` | `MixtureDataset`(소스별 비중, 새 데이터 약 4 Hz 행 1 epoch, 행별 손실 표시), `FtSelect`(500 step마다 검증 4종, 최종 선택), `choose`(선택 규칙) |
| `morai_vla/student.py` | 행별 손실 분기(`gt_row`), 이력 드롭아웃(`hist_drop`, `drop_history_tokens`). 두 키가 없는 배치는 전과 같다(테스트로 확인) |
| `morai_vla/data.py` | `MoraiDataset(keep=, exclude=)`: 행·에피소드 제한, 출발 재라벨(`src_i`), 행 제외(replay에서 재라벨 행). 빈 셀에서 `cell_keep`이 실패하던 버그 수정 |
| `morai_vla/early_stop.py` | `expert_ade`가 모델의 장치를 쓴다(전에는 `.cuda()` 고정; GPU 결과 불변) |
| `morai_vla/train.py` | `FtSelect.finalize`로 `final/`을 고른다 |
| `morai_vla/build_index.py` | `--ego`(R2 _ego 표 폴더), `--held-out`(기존 index_v3의 보류 에셋 그대로), `--glitch {step,net}`(10/28: R2는 `net`, 아래 '사용자 결정'). 기본값(`step`)은 그대로라 R1 index는 바뀌지 않는다 |
| `scripts/ft_synth_hist.py`, `assets/ft/synth_hist.parquet` | 10/28 합성 이력(차로 중앙선 모양, 실패하면 곧은 이력)과 D 출발 행. CPU 전처리, 결과 8 MB는 코드와 함께 상자로 간다. 확인 그림 `../reports/synth_hist/`(12장) |
| `scripts/ft_pools.py`, `assets/ft/` | 기존 데이터 풀(`old_recovery`, `old_curve`, `old_static`)과 R2 묶음별 에피소드 목록(`r2_A/B/D/AB`, 수집 뒤) |
| `scripts/ft_measure.py` | 측정: `history`(이력 의존도), `static`(정적 장애물). BF16 체크포인트와 5080 엔진 둘 다 |
| `scripts/pipeline_student.sh` | `ft` 단계, `gate_ft`, `QAD_FROM`(QAD 입력 단계, 기본 s2). 10/28: `qad` → `deliver` → `qgate`(관문은 납품 뒤, 결과는 `deliver/gate.json`) |
| `scripts/ft_metrics.py`, `tests/test_ft_metrics.py` | 메트릭 CSV·그림(10/28): `metrics/{train,eval,val}.csv`, `*.png`. 메트릭 동기화 루프와 ft 단계 끝에서 돈다 |
| `morai_vla/speed.py`, `scripts/ft_speed_probe.py`, `tests/test_speed.py`, `../tools/vast/r2ft_probe.sh` | 속도 최적화와 1-GPU 시험. **쓰지 않는다**(사용자 10/28 "그냥 안할래"). 코드는 기본 꺼짐으로 남아 있다 |
| `../tools/vast/r2ft_{stage,chain,speed,qadmini,watch}.sh` | 상자 스테이징, 지연 0 체인, 다운로드 중 속도 측정과 QAD 소형 시험(10/28), Mac 감시(아래 런북) |
| `tests/test_finetune.py`, `tests/ft_synth.py`, `configs/tiny/student_ft.yaml`, `scripts/ft_tiny.sh` | 단위 테스트 11개, 합성 step-0 루트, CPU 소형 실행 |

### 레시피 → 구현

| 레시피 | 구현 |
|---|---|
| 시작점 s2 final, 1단계 공동 학습 | `model.init=${paths.s2}`, `train_parts` 전체(카메라 ViT, LiDAR, LLM, LM head, expert). KD 투영(`kd_proj`)은 T* ViT KD 전용이라 10/28부터 얼린다 |
| 카메라 ViT 0.1배, expert 0.3–0.5배, LLM·LiDAR 1배 | `lr_multiplier {vlm.model.visual: 0.1, expert: 0.3, action_in_proj: 0.3, action_out_proj: 0.3}` |
| 최고 3e-6, 워밍업 약 200, 1e-6까지 cosine | `learning_rate 3e-6`, `warmup_steps 200`, `cosine_warmup_with_min_lr`, `min_lr 1e-6`. HF는 min_lr을 기본 LR 비율(1/3)로 바꿔 모든 그룹에 걸므로, 그룹마다 최고값의 1/3에서 끝난다 |
| 혼합 A 30 / B 15 / D 15 / 기존 복구·곡선 10 / 출발 재라벨 5 / replay 25 % | `data.train_dataset.sources`. 기존 복구·곡선 10 %는 복구 5 % + 곡선 5 %로 나눴다(아래) |
| 행 구성: 새 데이터를 2배 촘촘히(10/27 정정) | 새 A·B·D 행 = 2 Hz 행(`i % 5 == 0`) + 0.5 s 구간마다 0.2 또는 0.3 s 뒤 행 하나(에피소드·구간 해시, `extra_offsets [2, 3]`) ≈ 4 Hz. 한 epoch의 샘플 = 그 행 수 ÷ 0.60이고, 길이는 아래 2 epoch다. 소스마다 비중만큼 중복 없이 뽑는다. 기존 소스(복구·곡선·재라벨·replay)는 2 Hz 행에서만 뽑는다. replay의 2 Hz 행은 s2의 행 그대로다 |
| GT 행: FM + 궤적 CE (10/28 사용자: 정답만, T* 없음) | `paths.teacher: null`, `loss_weights` kv·vit·vkd·coc_kl 0. `gt_row=True` 행은 fm·traj_ce(lm 없음: R2에 CoC 라벨이 없다) |
| replay: FM + 궤적 CE + lm (10/28, 증류 없음) | `gt_row=False`(소스 종류 이름 `distill`은 코드 경로 때문에 그대로): fm, traj_ce, lm(끝 토큰; coc false는 s2와 같다). `kd_proj`는 T* ViT KD 전용이라 얼린다. 출발 재라벨된 (episode, i)는 replay에서 뺀다(`exclude`) |
| 이력 (10/28): 복구 행 실제·정지·합성 1/3씩, 다른 GT 행 실제·정지 50 %씩, D 출발 1.5 s 행은 합성 | 소스 설정 `hist: thirds`(new_d, old_recovery)와 `dataset.synth: ${paths.synth_hist}`(`assets/ft/synth_hist.parquet`, `scripts/ft_synth_hist.py`). 같은 blake2b 해시로 고른다(모든 rank·재개에서 같다). 합성 행은 `hist_synth`, `hist_xyz`/`hist_rot`로 이력 토큰만 바꾼다(`drop_history_tokens(hist=)`). D 출발 행은 `synth_only` 행이다: index에 없는 i < 15를 붙이고, 그 행의 이력 자체를 합성 이력으로 쓴다(토큰과 행동 공간 모두), 미래는 실제다. 시작 로그의 `hist_drop_share`, `hist_synth_share` |
| 길이 2 epoch (10/27 21시 사용자 "2에폭으로 가자") | `epochs: 2`: 전체 샘플 = 새 약 4 Hz 행 × 2 ÷ 0.60, step = 샘플 ÷ 32. 10/28 데이터 기준 추정 약 106,900 샘플 = **약 3,342 step**(계산은 런북 3단계) |
| 약 500 step마다 검증 4종(기록), 최종 = 마지막 체크포인트, 중단 스위치 = 횡오차 +100 % (10/27 "넣자", 10/28 "+25 %로 완화", 10/28 새벽 "중단 스위치를 횡방향 ADE 기준으로 바꿔. general_wo_relabel 횡오차가 s2의 2배(+100 %)를 2번 연속 넘으면 멈춤.") | `FtSelect(every_steps 500, ref general, rel_worse null, stop_ref general_wo_relabel, stop_exclude ${paths.start_relabel}, stop_metric lat, stop_rel_worse 1.0, stop_patience 2)`. step 0(= s2 final)을 기준으로 잰다. `general_wo_relabel`의 `stop_metric`(`ade`·`lat`·`fde`·`lon` 중 하나; 10/28부터 `lat`)이 기준보다 `stop_rel_worse` 넘게 나쁜 측정이 2번 연속이면 그 step에서 체크포인트를 남기고 멈춘다(`ft_stop.json`). 그러면 파이프라인 `gate_ft`가 QAD 전에 HALT한다. `stop_rel_worse: null`이면 꺼진다 |
| R2 index의 급변 필터 (10/28 "동적 장애물에 의한 급제동 데이터를 남겨야지") | R2 index만 `build_index --glitch net`: 0.1 s 속도 계단(MAX_DV)을 바로 버리지 않고 앞뒤 0.8 s 순변화로 본다. 순변화가 같은 방향이고 계단의 절반 이상이며 15 m/s² 이하면 실제 급제동·급가속으로 남긴다. GT 튐(되돌아오는 계단), 출발 과도(τ < 3.5 s), 15 m/s² 넘는 감속은 그대로 버린다. R1 index는 `step` 그대로다 |

### 정한 것

- **새 B처럼 비중보다 행이 적은 소스**: 묶음별 행 수가 정확히 2:1:1이 아니다(B 에피소드가 짧다). 비중을 지키려고 그 소스는 약 4 Hz 행을 다 쓴 뒤 아직 안 쓴 다른 격자 행(오프셋 1, 4)으로 채운다. 프레임이 달라 같은 샘플은 아니다. 모든 행을 쓰고도 모자라면 그때 반복한다. 행이 남는 소스(A 예상)는 무작위 일부만 쓴다. 소스별 실제 값은 시작 로그 `MixtureDataset: {...}`의 `rows_used`·`other_offsets`·`repeated`에 나온다(경고도 남는다).
- **해시**: 오프셋 선택과 드롭아웃 추첨은 blake2b를 쓴다. `data.unit_hash`(CRC-32)는 선형이라, 끝자리만 다른 키(한 에피소드의 연속 구간)에서 값이 몰렸다(시험에서 한 에피소드의 9구간이 모두 같은 쪽). LiDAR 드롭아웃 마스크는 교사와 맞춰야 하므로 `unit_hash` 그대로다.
- **expert 0.3배 (0.3–0.5 중)**: 기본 LR 3e-6 × 0.3 = 9e-7로, s2의 expert 최고 LR(1e-5 × 0.1 = 1e-6)과 거의 같다. 그 속도에서 s2가 안정적으로 돌았고 관문도 통과했다(QAD/s2 1.015). 새 행동(걷는 보행자 앞 정지, 복귀 곡률)은 장면 판독이 먼저라 1배로 배우는 LLM KV를 거쳐 expert에 들어간다. GT FM 그래디언트도 KV로 간다. 새 데이터는 서로 다른 행이 약 3–5만 개뿐이다. 그래서 2.3B expert를 더 빨리 움직일 이유가 약하고, 대신 옛 동작을 지키는 것이 25 % replay(정답 손실)뿐인 동안 잊을 위험이 있다. 동적 검증의 FM·ADE가 내려가지 않으면 0.5로 다시 돌리는 것이 대안이다(`FTEXTRA="trainer.lr_multiplier.expert=0.5 trainer.lr_multiplier.action_in_proj=0.5 trainer.lr_multiplier.action_out_proj=0.5"`).
- **이력 드롭아웃 = 0 변위 토큰**: 가린 행은 16점 이력을 원점에 서 있는 차(위치 0, 회전 단위행렬)로 바꿔 LLM 토큰만 다시 만든다. 미래 토큰과 나머지 위치는 그대로다.
  - 이것이 배포 엔진의 '이력 없는' 입력과 같다. e2e_driver는 이력 16점을 GT 속도·yaw를 `np.interp`로 격자에 맞춰 만드는데, 첫 기록보다 앞은 첫 값(정지)으로 채운다. 그래서 에피소드 시작 때 엔진이 만드는 입력이 0 변위 이력이다. 토큰 수와 위치가 같아서 엔진·프롬프트를 고칠 필요가 없다.
  - 행동 공간은 실제 이력을 그대로 쓴다. v0 추정, GT (a, κ) 목표 계산, 궤적 적분 모두 그렇다. 추론 때 엔진이 실제 이력으로 v0를 잡고 적분하는 것과 같다(테스트가 `_process_traj_future_training`이 실제 이력을 받는지 확인).
  - 새 토큰을 쓰는 마스크는 고르지 않았다. 학습에만 있는 입력이 되고, 엔진은 그 토큰을 만들지 않는다. 0 변위는 실제로 정지한 차와 구별되지 않는다. 그래서 '정지 이력 = 계속 정지'라는 지름길이 끊긴다(출발 대기 문제와 같은 방향).
- **기존 복구·곡선 10 % → 복구 5 % + 곡선 5 %**: index_v3 train 2 Hz에서 곡선 행은 89.2k, 복구 행은 14.9k다. 둘을 합쳐 고르게 뽑으면 복구가 그 칸의 15 %뿐이다. 이 칸은 주로 복구를 늘리려는 것이라 반씩 나눴다. 합은 표의 10 % 그대로다. 합쳐 뽑으려면 두 소스를 `old_recovery_curve` 하나(share 0.10)로 바꾸면 된다.
  - 복구 = 전역 경로에서 |d| ≥ 0.5 m 벗어나 있고 3 s 뒤 GT가 0.3 m 이상 가까워지는 행이다(v ≥ 2 m/s, C5 제외). step-0 표에는 노이즈 표시가 없어 R1 노이즈를 직접 고를 수 없다. 그래서 노이즈·스폰 오프셋·장애물 통과 뒤의 모든 복귀를 잡는다. 셀별로는 C6 5.6k, C3 3.4k, C2 3.4k, C0 2.5k다.
  - 곡선 = 3 s 동안 GT 방향 변화 15° 이상(v ≥ 3 m/s)이거나, 밀집 주행에서 이탈한 곡선(s 83–116, 193, 439, 594, 682 ±15 m, v ≥ 2 m/s)이다.
- **검증 4종** (`callbacks.ft_select.sets`):
  - 일반: 관문·s1 조기 종료와 같은 1,000행(val + val_asset, stride 10, seed 1)이다. 최종 선택과 관문의 기준이다. 이 중 약 9행(0.9 %)은 출발 재라벨 대상 행이라 정답이 옛 '기다림'이다. 그래서 중단 스위치는 그 행을 뺀 `general_wo_relabel`을 본다(사용자 10/28 "ㅇㅋ", `ft_select.stop_ref`·`stop_exclude`). 측정마다 둘 다 기록한다.
  - 동적: R2 A+B의 보류 장면 300행.
  - 정적: 기존 C2 보류 300행.
  - 곡선: 기존 곡선 보류 300행 + R2 D 보류 300행. 실제로는 R2 D 보류 장면이 670행(stride 10에서 66행, 쓸 수 있는 행 약 47)뿐이라 곡선 세트는 약 347행이다(10/28 검토).
  - 측정마다 세트별 ADE·FDE·횡오차·종오차(GT 진행 방향 기준 |오차| 평균)를 `ft_select.json`의 `metrics`에, 행별 값을 `ft_val_rows.jsonl`에 남긴다.
  - 지표는 `early_stop.expert_ade`(6.4 s, K=6, LiDAR 켬, 샘플마다 seed 0)다. 측정 1번은 8장에서 약 2–3분이다.
  - 측정마다 학습 파라미터 스냅샷(`ft_snapshots/`, BF16 약 14 GB)을 남긴다. 학습 끝 측정까지 약 7–11번이고, 합 약 100–160 GB다.
  - `ft_choice.json`에 고른 step(= 마지막 측정)과 기준값이 남는다.
- **HF eval** (사용자 10/28 "켜"): base의 기존 val 512행을 replay 방식(fm·traj_ce·lm)으로 500 step마다 잰다. `eval_fm`·`eval_traj_ce` 곡선을 보는 용도다. `gate_ft`는 NaN·inf만 본다. final/은 이미 위 규칙이 골랐다.
- **QAD (10/28 사용자 결정)**: `configs/student_qad_r2.yaml`을 쓴다. ft와 같은 혼합과 정답 손실이고, 옛 '기다림' 행을 뺀다. T*는 쓰지 않는다.
  - `student_qad`의 4.75 bits AutoQuantize, 800 step, lr 1e-6(모든 파트 같은 배율), warmup 50, final/ = 가장 낮은 eval_fm은 그대로다.
  - `qad_search.py`는 보정 샘플 128개를 섞인 schedule에서 고르게 뽑는다.
  - 관문(`gate_student.py --config-name student_ft`)은 학생만 잰다(T* 없음, 같은 val 500행, seed = 행 번호). 규칙은 `ratio_vs_s2` < 1.25(QAD / ft final) 그대로다.
  - 기록용으로 `general_wo_relabel`(관문 행에서 재라벨 행 제외)과 동적·곡선·출발 재라벨 세트 ADE를 ft final과 QAD 양쪽에서 남긴다. 세트 기록은 관문과 나란히 GPU 1–6에서 돈다. 실패해도 관문은 그대로다(`summary.json`의 `records`).

### 사용자 결정 (10/28, 이 run의 최종값)

| # | 결정 | 사용자 말 | 지금 값 (구현) |
|---|---|---|---|
| 1 | 중단 스위치 기준 세트 | "ㅇㅋ" | `general_wo_relabel`(일반 1,000행 − 출발 재라벨 행 약 9개). 관문·최종 선택은 `general` 그대로 (`ft_select.stop_ref`, `stop_exclude`) |
| 2 | R1 index(누수 0.43 %) | "ㅇㅇ 둬" | 다시 만들지 않는다 |
| 3 | 2차 녹화분(r2g·r2h·r2j·r2k…) | "ㅇㅇ 빼" | 제외. `assets/ft/r2_*.parquet`가 1차 892개만 담고, `ft_pools.py r2`를 다시 돌리지 않는다 |
| 4 | 곡선 검증 세트 | "둬" | 그대로(실제 약 347행) |
| 5 | HF eval | "켜" | 켬(500 step마다) |
| 6 | `torch.compile` | "쓰지말어 그럼" | 쓰지 않는다 |
| 7 | 움직이는 소품만 있는 A/B 에피소드 | "ㅇㅇ" | 학습 풀에서 제외(A 481 → 396, B 176 → 160, AB 657 → 556; 목록 `pools.json`의 `r2_drop_props`) |
| 8 | MAX_DV 확인 | "확인해봐" | 분석 뒤 15번으로 R2 규칙을 바꿨다 |
| 9 | 채택 | "일 좀 복잡하게 만들지 마. 그냥 하던대로 해" | QAD → 납품 → 관문 → 폐루프 확인, 비교 실험을 더하지 않는다 |
| 10 | 속도 최적화 | "그냥 안할래" | 쓰지 않는다: `FAST=0`(기본), 다운로드 중 최적화 검증·배치 2·1-GPU 시험 없음. GPU당 배치 1 × 누적 4 그대로 |
| 11 | 손실 | "ft와 QAD 모두 하드 타겟으로 바꿔. T* 안 쓰고, QAD도 ft와 같은 데이터 혼합(옛 기다림 행 제외)으로." | `paths.teacher: null`, fm·traj_ce(·replay lm)만. QAD = `student_qad_r2`(ft 혼합, replay `exclude`). 상자는 t2·rule_ok를 받지 않는다 |
| 12 | 납품·관문 순서 | "납품이랑 관문 순서가 반대야" | `STAGES="ft qad deliver qgate"`: QAD 직후 납품, 관문 결과는 `deliver/gate.json`. 관문 실패는 납품본을 지우지 않는다 |
| 13 | 이력·D 출발·중단 스위치 | "추천안대로 진행해. 이력은 전처리에서 차로 중앙선 모양(실패하면 곧은 이력)으로 만들고, 복구 행은 실제·정지·합성 1/3씩, D 출발 1.5초는 합성 이력으로 넣고, 중단 스위치는 +25 %로 완화." | `hist: thirds`(new_d, old_recovery), `synth_hist.parquet`, D 출발 행 847개(`extra`), `stop_rel_worse 0.25` |
| 14 | 다운로드 중 QAD 소형 시험 | "이게 좋겠어" | `tools/vast/r2ft_qadmini.sh`: GPU 0, 실제 s2, R2 데이터만, 보정 8 → 학습 5 step → manifest → 납품 → 관문·기록 8행. 결과 `out/qadmini/RESULT`, B2 `runs/r2ft/qadmini/<RUN>/`. ft 시작을 막지 않는다 |
| 15 | R2 급변 필터 | "ㅇㅇ 동적 장애물에 의한 급제동 데이터를 남겨야지" | R2 index만 `--glitch net`(위 레시피 표). B2 `step0_r2/_index`를 교체했고, 옛 파일은 B2 안에 `*.bak-20261028`로 있다 |
| 16 | 중단 스위치 (13번의 ADE +25 %를 대체) | "중단 스위치를 횡방향 ADE 기준으로 바꿔. general_wo_relabel 횡오차가 s2의 2배(+100 %)를 2번 연속 넘으면 멈춤." (그 전 "중단 스위치 끄고 끝까지 학습해. 폐루프로 판단할게."는 이것으로 대체) | `stop_metric: lat`, `stop_rel_worse: 1.0`, 기준 0.230 m → 한계 0.460 m. 도는 H100 run은 상자의 `resume_nostop.sh`가 새 스위치로 잇는다(옛 스위치가 step 1000에서 걸리면 그 checkpoint에서, 안 걸리면 checkpoint-1000에서). 감시는 `--stop-off`로 옛 스위치의 HALT만 경보에서 빼고, 새 스위치가 걸리면 경보를 낸다 |

**7. 소품 기준:** `_meta/r2_episodes.parquet`에서 속도가 0보다 큰 객체가 모두 소품(에셋 id 4xxxxxxx)인 에피소드다. 소품은 차량형 소품, 무인 자전거·스쿠터·유모차다. 사람·동물·사람이 탄 자전거(5xxxxxxx)가 하나라도 있으면 남긴다. MORAI 10/6 답변: 본선에서 정적 장애물은 움직이지 않는다. 명령은 `python scripts/ft_pools.py drop-props --meta <step0_r2>/_meta/r2_episodes.parquet --out assets/ft`. 혼합 비율(A 30 / B 15 %)은 그대로이고, 소품과 사람이 섞인 A 60개, B 17개는 남는다.

**13. 합성 이력:** `scripts/ft_synth_hist.py`(CPU)가 1.5 s 이력을 경로 중앙선에 평행하게, 마지막 1.0 s에 걸쳐 현재 heading으로 이어지게 만든다. |횡오프셋| > 4.5 m, |방향 오차| > 30°, 투영 위치가 진행거리와 5 m 넘게 다름, 경로 시작 전이면 곧은 이력이다. 15번 index로 다시 만든 결과:

| 대상 | 행 | 중앙선 방식 | 곧은 이력 |
|---|---|---|---|
| R2 D (train) | 20,039 | 18,527 | 1,512 |
| D 출발 i < 15 (train, 75개 에피소드) | 847 | 777 | 70 |
| 기존 복구 (2 Hz train) | 14,924 | 13,944 | 980 |

**15. 급변 필터 결과:** R2 index 111,935 → 121,523행(+9,588). 버리는 행 12,764 → 3,176. 묶음별(전체/train): A +5,109/+4,683, B +2,140/+1,819, D +839/+759. 늘어난 행 중 4,413행이 AEB·양보 기록과 겹친다. 쓸 수 있는 새 4 Hz 행은 약 +2,352라 step은 약 3,097 → 3,342다.

### 새 데이터 경로 채우기 (수집이 끝난 뒤)

1. **step-0** (winpc CPU, tools/step0 README): R2 bag(`D:/share/2026_MORAI_collect_R2_20261027`) → 새 루트(예: `D:/share/2026_MORAI_step0_R2`)에 에피소드별 `cams.bin`, `cams_index.npz`, `lidar.bin`, `lidar_index.npz`와 `_ego/<ep>.npz`를 만든다. 기존 step-0 루트와 섞지 않는다.
2. **R2 index** (Mac): R2 `_ego/*.npz`를 Mac의 별도 폴더(예: `~/.cache/step0_ego_r2`)로 복사한다. 그다음 기존 index_v3와 같은 보류 에셋으로 만든다.
   ```bash
   uv run --with pandas,pyarrow,pyyaml,numpy,scipy python -m morai_vla.build_index ~/coc_runs/index_r2 --ego ~/.cache/step0_ego_r2 --held-out 40100027,40100029,50100001 --glitch net
   ```
   `index.parquet`를 R2 루트 `_index/`에 넣는다. `--glitch net`은 10/28 결정 15번이다(B2 `step0_r2/_index`는 이 규칙으로 교체됨, 121,523행). 그다음 합성 이력을 다시 만든다: `python scripts/ft_synth_hist.py --r2-ego ~/.cache/step0_ego_r2 --r2-index <R2 index> --r1-ego ~/.cache/step0_ego --r1-index <R1 index> --out assets/ft/synth_hist.parquet --figs ../reports/synth_hist`. 라벨(labels-cache)과 매니페스트(rv*)는 기존 경로에서 읽는다. 보류 에셋을 고정하지 않으면 R2 매니페스트의 새 에셋 때문에 보류 목록이 바뀐다.
3. **R2 묶음 목록** (Mac): QA 파일(`<수집 폴더>/qa.jsonl`, `ok`가 참인 에피소드)과 함께 만든다. 파일럿(r2p*)은 빠진다.
   ```bash
   python scripts/ft_pools.py r2 --kinds ~/.claude-morai-mac/episode-kinds.tsv --qa <각 수집 폴더>/qa.jsonl --out assets/ft
   ```
   `assets/ft/r2_{A,B,D,AB}.parquet`가 생긴다. 기존 풀(`old_*`)은 10/27에 index_v3 + `~/.cache/step0_ego`로 이미 만들었다(`assets/ft/pools.json`).
4. **출발 재라벨** (확정, 사용자 승인 10/27): step-0 미러의 `_start_relabel/start_relabel_index.parquet`이다(B2 `step0/_start_relabel/`, winpc `D:/share/2026_MORAI_step0/_start_relabel/`). 기본값이 `${data.root}/_start_relabel/start_relabel_index.parquet`이라, `/data/step0`로 받으면 따로 줄 것이 없다. 54,285행(10 Hz)이고, index_v3 기준이며, 항상 src_i > i다(당긴 시간 중앙값 2.9 s, 최대 9.6 s). 같은 폴더 README.md에 규칙과 스키마가 있다.
   - 열은 `episode, i, src_i`(+ 참고 열)다. 행 i는 자기 입력을 쓰고 미래만 같은 에피소드 행 src_i(출발 시점, ≥ i)의 것이다.
   - i와 src_i가 모두 index의 학습 가능 행이어야 하고, 아니면 버리고 경고한다(재라벨 세션 점검: 614행이 빠지고 train 50,555행).
   - 같은 파일이 replay의 `exclude`로도 쓰인다(`paths.start_relabel` 하나).
5. **학습 상자**: B2 `morai2026-alpamayo`에서 받는 것은 `tools/vast/r2ft_stage.sh`가 정한다(아래 런북 1단계). R2 실행의 `B2`는 `runs/r2ft`라 pipeline의 `need`가 s2를 찾지 못하므로, 스테이징이 `runs/stageC/s2/final`을 `$OUT` 아래에 미리 받는다(10/28: T* 없음이라 t2·rule_ok는 받지 않는다).

### 런북 (10/28 검토판): 빌리기 전 → 상자 → 납품

명령은 Mac 워크스페이스 루트(`Alpamayo-Cosmos/`)에서 복사해 그대로 쓴다. 상자 스크립트는 10/27 텍사스 상자(B2 `runs/r2ft/pro6000tx_57966802/`)에만 있던 것을 저장소로 복원하고 고쳤다.

| 스크립트 (`tools/vast/`) | 하는 일 |
|---|---|
| `r2ft_stage.sh` | 도구 설치 → 감시견 1개 + 메트릭 동기화(10분, CSV·그림 포함) → 병렬 받기: R1 step0, R2 step0, s2(10/28: T* 없음이라 t2·rule_ok는 받지 않는다), venv(GPU에 맞게 자동 선택) → 학생 패키지 → `gpu_check`, HF Qwen3-VL 4B(+8B 설정 파일). 단계마다 `stage.log`에 `<시각> <단계> rc=<n>` |
| `r2ft_chain.sh` | R1을 뺀 나머지가 오면 QAD 소형 시험(GPU 0)과 속도 측정(GPU 1–7)을 함께 시작 → R1이 끝나는 순간 둘을 멈추고(소형 시험이 안 끝났으면 `INCOMPLETE`로 기록), 나머지 단계가 모두 rc=0인지 보고, `pipeline.sh STAGES="ft qad deliver qgate"`를 바로 시작한다. step 0은 ft_select가 그 run에서 잰다(10/27 H200 값에는 `general_wo_relabel`이 없다). 어텐션은 GPU로 자동(sm_12x = RTX PRO 6000/5090 → sdpa, 그 밖 → flash_attention_2) |
| `r2ft_qadmini.sh` | 다운로드 중 QAD 소형 시험(10/28 결정 14): 실제 s2, R2 데이터만, `qad_search`(보정 8) → QAD 5 step(LiDAR ViT만 학습: GPU 1장에 6.5B의 Adam 상태가 들어가지 않는다) → manifest → 납품 → 관문·기록 8행 → `gate.json`. 약 10–15분. `out/qadmini/RESULT` = `PASS <초>` / `FAIL <단계>` / `INCOMPLETE ...`, 로그·JSON은 B2 `runs/r2ft/qadmini/<RUN>/`(가중치 제외) |
| `r2ft_speed.sh` | 다운로드 중 50 step 속도 측정(실제 ft 코드 경로, R2 데이터만, GPU 1–7). `out/speed/SPEED.json`(s/step 중앙값, 최대 메모리)과 `sys.log`(GPU 사용률·전력·클럭, CPU MHz, 프로세스 CPU %) |
| `r2ft_watch.sh` | Mac에서 2분마다 ssh로 보고, 실패 상태에서 한 줄을 내고 끝난다(아래 4단계) |

**0. 빌리기 전 (Mac, 무료)**
- 조건: GPU 8장 × 80 GB 이상, 디스크 1 TB 이상, 회선 1 Gbps 이상, CPU 단일 코어가 빠른 상자.
  - GPU당 최대 메모리: 10/27 실측은 T*를 실은 상태로 67–68 GB였다. 10/28부터 T*가 없어 약 40 GB로 추정한다(T* 가중치 약 28 GB와 그 forward가 빠진다).
  - CPU 디스패치가 속도를 정한다(H200 프랑스 8 s/step, PRO 6000 펜실베이니아 6.15 s/step 모두 CPU 병목. 기준 s2 이탈리아 2.75 s/step, T* 포함).
- 재고 검색은 세 곳 모두 한다.
  ```bash
  for g in H200 H200_NVL H100_SXM RTX_PRO_6000_S RTX_PRO_6000_WS B200; do vastai search offers "num_gpus=8 gpu_name=$g rentable=true disk_space>=1000" -o dph_total | head -6; done
  bash tools/vast/runpod_stock.sh 8 "NVIDIA H200" "NVIDIA H100 80GB HBM3" "NVIDIA RTX PRO 6000 Blackwell Server Edition"
  for g in H200 H100 RTX_PRO_6000; do brev search gpu --gpu-name $g | grep -E " 8 "; done
  ```
- 사용자에게 표로 보이고 승인을 받은 뒤 빌린다. 열: GPU, $/h, CPU 모델·클럭, 회선, 신뢰도, 디스크, **총비용** = $/h × (부팅 + 받기 + 학습 + 측정 + QAD·관문·납품).
  - 받기 약 238 GB: R1 step0 약 190 GB, R2 16.1 GB, s2 15.1 GB, venv 약 8 GB, Qwen3-VL-4B 8.9 GB. 10/28에 T*(t2 27.9 GB)가 빠졌다. B2 실측 약 130 MB/s(H200 프랑스: R1 190 GB 25분) → 1 Gbps 이상이면 약 30–35분, 544 Mbps(텍사스)면 65–75분. R1이 가장 오래 걸리는 것은 그대로다.
  - 학습 약 3,342 step(10/28: 소품 에피소드 제외, D 출발 행, R2 급변 필터 반영) × s/step. T*가 빠져 step이 빨라진다(추정 20–35 %). 좋은 CPU의 H200이면 2.75 s → 약 1.8–2.2 s, 약 1.7–2.0 h. 느린 CPU면 약 5–6 s, 4.6–5.6 h. 측정 약 8번(step 0 포함) × 3–7분 ≈ 0.4–0.9 h(학생 추론이라 변화 없음). QAD 약 0.6–0.7 h(800 step × 약 2.3–3.0 s + 탐색). 관문 약 15분(학생만). 납품 3분.
  - vast의 `cpu_ghz`는 광고값이다. 프랑스 H200의 Xeon 8568Y+는 4.0으로 나왔지만 2.3 GHz에 고정돼 있었다. 그래서 2단계 인수 시험이 최종 판단이다. Brev는 CPU 클럭과 회선을 아예 보여 주지 않는다.
- 제공자별 준비:
  - vast: 계정 환경변수 `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`(빌리기 전에), 이미지 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`.
  - RunPod: 같은 두 키를 secret으로, `DPH_USD`(감시견이 쓴다).
  - Brev: `brev login`은 사용자가 한다. 컨테이너 환경이 없으므로 상자에 `/workspace/b2.env`(두 키)를 직접 만든다. 감시견이 상자를 스스로 멈추지 못하므로 담당 세션이 `brev stop`으로 멈춘다.

**1. 빌린 직후 (Mac): 코드 → 스테이징 → 체인, 한 번에**
```bash
ID=<instance id>; RUN=<gpu>_$ID; RENTED=$(date +%s)          # 예: RUN=pro6000_57999999 (B2 runs/r2ft/metrics/$RUN)
url=$(vastai ssh-url $ID); hp=${url#ssh://}; HP="${hp%:*} -p ${hp##*:}"
COPYFILE_DISABLE=1 tar czf - --exclude __pycache__ --exclude '*.bak-*' --exclude train/outputs --exclude train/logs train tools/vast \
  | ssh -o StrictHostKeyChecking=no $HP 'mkdir -p /workspace && tar xzf - -C /workspace'
ssh $HP "RUN=$RUN setsid nohup bash /workspace/tools/vast/r2ft_stage.sh > /workspace/stage.log 2>&1 < /dev/null &
         setsid nohup bash /workspace/tools/vast/r2ft_chain.sh > /workspace/chain.log 2>&1 < /dev/null &"
```
- 사용자 10/28: 속도 최적화는 쓰지 않는다. 그래서 `FAST`를 주지 않는다(기본 0, 기존 검증된 동작).
- 코드 묶음은 약 1.7 MB다. 이 뒤로 사람이나 에이전트가 끼지 않는다. R1이 끝나는 순간 체인이 본 학습을 시작한다.

**2. 인수 시험 (빌린 뒤 10–30분, 다운로드와 동시)**
```bash
ssh $HP 'grep -E "rc=|gpu check|run " /workspace/stage.log; cat /workspace/chain.log; du -sh /data/step0 /data/step0_r2 2>/dev/null; cat /workspace/out/speed/SPEED.json 2>/dev/null; tail -12 /workspace/out/speed/sys.log 2>/dev/null; cat /workspace/out/qadmini.log /workspace/out/qadmini/RESULT 2>/dev/null'
```
- 통과 조건:
  - `gpu check PASS`
  - B2 받기 약 100 MB/s 이상(`du -sh /data/step0`를 몇 분 간격으로 두 번)
  - 속도 측정 s/step이 기준의 1.2배 이하(H200 3.3 s; RTX PRO 6000은 빠른 CPU 상자 실측이 아직 없다. 펜실베이니아 6.15 s는 CPU 병목 값이다)
  - 부하 중 CPU MHz가 터보 근처(2.3 GHz 고정이 아님), GPU 전력이 높음
- QAD 소형 시험은 상자 인수 조건이 아니다. `FAIL <단계>`면 그 단계 로그(`out/qadmini/<단계>.log`)로 고쳐 두고 본 학습은 그대로 둔다. QAD는 ft 뒤(약 2–3 h 뒤)에 돈다. `INCOMPLETE`는 R1이 먼저 와서 멈춘 것이다.
- 실패하면 바로 지운다(메트릭을 먼저 올린다): `bash tools/vast/safe_destroy.sh $ID /workspace/out/stageR2 b2://morai2026-alpamayo/runs/r2ft/metrics/$RUN`
- 30분 손절: 빌린 지 30분 안에 속도 측정 step이 기대 속도로 나오지 않으면 지우고 보고한다. 허락 없이 상자를 겹쳐 돌리지 않는다.

**3. 본 학습 (자동, 지연 0) — 시작 5분 안에 상자에서 직접 확인**
```bash
ssh $HP 'tail -4 /workspace/chain.log; grep -m1 "MixtureDataset:" /workspace/out/stageR2/logs/ft.log | cut -c1-300; grep -m1 "exclude " /workspace/out/stageR2/logs/ft.log; tail -2 /workspace/out/stageR2/ft/loss_components.jsonl; nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader'
```
- 기대값(추정): 10/27 H200 시작 로그(`new_rows_4hz` 33,639)에 10/28 변경을 쓸 수 있는 행 비율(0.805)로 더했다: 소품 에피소드 제외 약 −4,115, D 출발 행 약 +200, R2 급변 필터 약 +2,352. `"total"` 약 106,900, `"epochs": 2`, `"new_rows_4hz"` 약 32,100, `gt_share 0.75`, 약 3,342 step이다. `hist_drop_share`·`hist_synth_share`가 함께 나오고, `exclude ... 53671 of 2086127 index rows left out`, `synth assets/ft/synth_hist.parquet: 20886 histories, 847 extra rows`(R2 D)와 `14924 histories, 0 extra rows`(기존 복구)가 rank마다 찍힌다. 10/28 H100 실측은 `total` 104,030, `new_rows_4hz` 31,209, 3,251 step, `hist_drop_share` 0.342, `hist_synth_share` 0.067이었다.
- `chain.log`의 "R1 step0 sync done"과 "pipeline start" 사이가 몇 초 이내여야 한다.
- 시작 시각, 총 step, ETA, 예상 비용을 바로 사용자에게 보고한다.

**4. 감시 (담당 세션이 직접, 이벤트마다 다시 띄운다)**
```bash
HP="$HP" RENTED=$RENTED EXPECT=2.75 bash tools/vast/r2ft_watch.sh     # EXPECT: 인수 시험에서 잰 s/step
```
- ALERT: 3번 연속 연결 안 됨, 빌린 뒤 30분까지 step 없음, HALT, 중단 스위치, 파이프라인 프로세스 없음, 로그의 Traceback·NCCL·OOM, s/step > 1.2 × EXPECT, GPU 사용률 20 % 미만 10분.
- STAGE: 본 학습 시작, 단계 시작·끝, `PIPELINE_DONE`, QAD 소형 시험 PASS.
- ALERT(10/28 추가): QAD 소형 시험 `FAIL`·`INCOMPLETE`. 사용자에게 알리고, ft는 계속 돈다.

**5. 메트릭 (자동)**
- 10분마다 `b2://morai2026-alpamayo/runs/r2ft/metrics/$RUN/`로 동기화하고, 상자가 멈추기 전에도 한 번 더 한다.
  - `ft/loss_components.jsonl`: 10 step마다 HF loss·grad_norm·learning_rate·s_per_step, 손실별 값(전 rank 평균), `<소스>/<손실>`(new_a, new_b, new_d, old_recovery, old_curve, start_relabel, replay)
  - `ft/ft_select.json`, `ft/ft_val_rows.jsonl`: 500 step마다 검증 4종 × ADE·FDE·횡오차·종오차, 행별 값
  - `ft/metrics/{train,eval,val}.csv`, `train_losses.png`, `train_sources.png`, `grad_lr.png`, `val.png`(`general_wo_relabel` +25 % 중단선 포함). 검증 세트 행에는 `general_wo_relabel`도 들어간다
  - `logs/`, `stage.log`·`chain.log`는 상자에 남고, `pipeline.sh`가 단계마다 B2 `runs/r2ft/<단계>`에 올린다.
- 보기: B2 키가 있는 곳(상자, winpc `~/b2tool/bin/b2`)에서 `b2 ls -r b2://morai2026-alpamayo/runs/r2ft/metrics/$RUN/ft/metrics/`. Mac에는 인증된 b2가 없다(10/28: `uvx b2` + 키 환경변수로 읽기만 했다).

**6. 중단 스위치·실패**
- 중단 스위치(10/28부터 `general_wo_relabel` **횡오차** > s2 final × 2가 2번 연속; 지표는 `stop_metric`, 한계는 `stop_rel_worse`) → 그 step의 체크포인트와 `ft_stop.json` → `gate_ft` HALT → 파이프라인 종료 → 감시견이 20분 뒤 상자를 멈춘다(디스크 유지). 담당 세션은 디스코드로 사용자에게 알린다.
- 버그였으면 고치고, 마지막 정상 체크포인트(멈춘 step − 1,000; `save_total_limit 3`이라 남아 있다)부터 잇는다.
  ```bash
  ssh $HP 'cd /workspace/out/stageR2 && mkdir -p stopped && mv ft/ft_stop.json ft/checkpoint-<멈춘 step> ft/checkpoint-<멈춘 step − 500> stopped/ && rm -f HALT'
  ssh $HP 'cd /workspace/train && OUT=/workspace/out/stageR2 B2=b2://morai2026-alpamayo/runs/r2ft STAGES="ft qad deliver qgate" QAD_FROM=ft R2ROOT=/data/step0_r2 SEXTRA="model.attn_implementation=<sdpa|flash_attention_2>" setsid nohup bash scripts/pipeline.sh > /workspace/out/pipeline.log 2>&1 < /dev/null &'
  ```
  `resume=true`가 남은 최신 체크포인트를 잡고, `FtSelect`는 그 뒤의 측정을 버린다.
- 레시피 문제로 보이면 `vastai stop instance $ID` 후 디스코드로 알린다.

**7. QAD → 바로 납품 → 관문 (자동, 사용자 10/27 규칙, 10/28 순서 정정)**
- `qad`: `qad_search`(GPU 1장, 4.75 bits) → QAD 800 step → `quant_manifest.json` → **바로 납품 export·B2 push**(사용자 10/28 "QAD되자마자 납품"; qad 자체의 약 30 GB push보다 먼저, 12 MB/s 상자에서 약 40분 빠름). 이 단계의 관문은 NaN·발산만 본다.
- `deliver`: 같은 `deliver/final` + manifest를 다시 만들고 올린다. 가중치는 mtime이 같아 b2 sync가 건너뛴다. 엔진 변환은 `qad` 단계에서 올라간 납품본으로 이미 시작할 수 있다.
- `qgate`: `gate_s2`(= ft final)·`gate_qad` 500행 → `ratio_vs_s2`.
  - 결과(`qgate/summary.json`, `pass`, 기록)를 `deliver/gate.json`으로 붙이고 deliver를 다시 올린다.
  - 1.25 이상이면 HALT하지만 납품본은 지우지 않는다. `gate.json`의 `pass: false`로만 보인다.
  - 10/28 사용자 결정: `student_qad_r2`(ft 혼합, 정답 손실, T* 없음)를 쓰고, 관문은 학생만 잰다.
  - 기록 세트(동적·곡선·출발 재라벨)는 관문과 나란히 GPU 1–6에서 돈다(`records`).
- 결과는 B2 `runs/r2ft/{ft,qad,deliver,qgate,done}`에 있고, 끝나면 `PIPELINE_DONE`이 생긴다.

**8. 끝**
```bash
bash tools/vast/safe_destroy.sh $ID /workspace/out/stageR2 b2://morai2026-alpamayo/runs/r2ft/metrics/$RUN
```
- 그 뒤 winpc에서 기존 방식으로 엔진(`~/morai-eval/engine/`)으로 바꾸고 폐루프를 확인한다(호스트 사용은 사용자 허락 뒤).

**설정 메모**
- 학습 설정은 레시피 그대로다: GPU당 배치 1 × 누적 4, ZeRO-2, GC 끔, 8장 × 4 = 32 샘플/step.
- 어텐션: H100/H200/B200은 `flash_attention_2`(base.yaml 기본), RTX PRO 6000·5090(sm_120)은 flash-attn 커널이 없어 `sdpa`다. `SEXTRA="model.attn_implementation=sdpa"` 한 줄로 학생·교사·QAD·관문이 모두 바뀐다(교사는 학생 설정의 값을 따른다). 추론(`sample_trajectories`)의 expert는 항상 sdpa다.
- `pipeline.sh`가 `ulimit -n`, `NCCL_NVLS_ENABLE=0`, `expandable_segments`를 건다.
- 수동 실행은 `pipeline_student.sh`의 `ft()`와 같은 명령이다.
  ```bash
  torchrun --nproc_per_node 8 -m morai_vla.train --config-name student_ft data.root=/data/step0 \
    paths.s2=$OUT/s2/final paths.r2_root=/data/step0_r2 paths.output_dir=$OUT/ft
  ```

### R2 속도 최적화: 쓰지 않음 (사용자 10/28 "그냥 안할래")

체인은 `FAST=0`(기본값)으로 띄운다. 다운로드 중 최적화 검증, 배치 2, 1-GPU 시험(`r2ft_probe.sh`)도 하지 않는다. `morai_vla/speed.py`, `scripts/ft_speed_probe.py`, `tests/test_speed.py`, `../tools/vast/r2ft_probe.sh`는 기본 꺼짐으로 남겨 두었고, 이번 run의 동작은 바꾸지 않는다. 그때 분석(항목·추정 절감·수학이 바뀌는 항목)은 `../reports/r2ft_prerent_review_20261028.md`의 '속도 최적화' 절에 있다. 그 절의 시간은 T*가 있던 10/27 기준이다. 스테이징 버그 수정(CoC·1.5 release를 받지 않기 등)은 FAST와 별개라 그대로 쓴다.

### 측정 (미세조정 전후, `scripts/ft_measure.py`)

```bash
# BF16: s2 final(전) / ft final(후)
python scripts/ft_measure.py history --root /data/step0 --ckpt $OUT/s2/final --out $OUT/measure/s2_history --n 300
python scripts/ft_measure.py static  --root /data/step0 --ckpt $OUT/ft/final --out $OUT/measure/ft_static
# 5080 배포 엔진 (winpc, eval 레포 서버 venv, MORAI_VLA_TRAIN=<이 train>): wowmayo_final(전) / 새 납품(후)
python scripts/ft_measure.py history --root <step-0 루트> --engine ~/morai-eval/engine/wowmayo_final --server <eval 레포>/src/e2e_driver/server --out <dir>
```
- **history**:
  - 조건은 세 가지다. base, swap(행 k가 행 (k + n/2) % n의 이력 토큰을 읽음; v0와 적분은 자기 이력), nocam(카메라 16장 검정, LiDAR 유지).
  - 조건마다 ADE, FDE, 1/2/3 s 횡오차를 GT 진행 방향 기준으로 잰다. base 대비 변화로 `*_move`(궤적 간 평균 거리)와 `*_dlat_2s/3s`도 잰다.
  - 이력 의존도는 `swap_move`, 카메라 의존도는 `nocam_move`로 본다.
- **static**:
  - 기본 대상은 기존 C2 보류 행(stride 5, 600행)이다.
  - GT 차체가 6.4 s 안에 정적 장애물 상자(`assets/rule_context.npz`) 3 m 안을 지나는 행에서, 예측·GT 차체 간격(평균, p10, 0.3 m 미만 비율, 겹침)과 ADE를 잰다.
  - 모든 행에서 6.4 s 평균 속도(예측 대 GT)도 잰다.
- 행마다 모든 조건에서 같은 초기 노이즈를 쓴다(seed = 행 위치). `--k`로 샘플 수를 늘린다.
- 엔진은 첫 호출에 컴파일·캡처로 몇 분이 걸린다. **엔진 경로는 GPU가 없어 아직 돌려 보지 않았다.** BF16 경로는 CPU 소형 모델로 돌렸다.

### 사전 검토 (Mac, 10/28 0시, 유료 상자 없이)

- `pytest tests` 70 passed, 1 skipped(NVFP4 fake quant, CUDA 필요). 전에 건너뛴 3개(행별 손실·이력 드롭아웃)는 HF 캐시에 Qwen3-VL-8B `config.json`과 1.5 release `config.json`만 받아 실제로 돌렸다. 메트릭 코드를 고치기 전 결과다. 고친 뒤에는 `ft_tiny.sh`(아래)와 새 `tests/test_ft_metrics.py`만 돌렸다.
- `scripts/ft_tiny.sh` 종단 통과(배치 2로 GT·replay 섞인 배치, 고친 메트릭 코드 포함): `loss_components.jsonl`에 `grad_norm`, `learning_rate`, `<소스>/<손실>`, `ft_select.json`에 세트별 ADE·FDE·횡·종오차, `ft_val_rows.jsonl`, `ft_metrics.py` CSV·그림.
- 중단 스위치: 같은 tiny에서 `stop_rel_worse=-0.5`로 측정마다 '나쁨'을 만들자 step 8(4·8 두 번 연속)에서 멈추고 `checkpoint-8`, `final/`, `ft_stop.json`이 남았다. `gate_ft`는 이 실행에서 "ft stop switch at step 8"로 HALT, 정상 실행에서는 통과했다.
- 실데이터(B2 10/27): 8×H200 시작 로그(B2 `runs/r2ft/h200_57960326`)와 index 대조. exclude 53,671행이 빠지고(s2 replay 2 Hz 행 중 7,907행), 재라벨은 54,285행 중 53,671행이 index에 있고 src_i는 모두 있다.
- 속도 측정·체인의 Hydra 오버라이드(소스 삭제, `sets.general.0.n=16`)는 compose로 확인했다.

### 로컬 검증 (Mac CPU, 10/27 17:30, venv `~/claude-bench/venv-rl`)

- `python -m pytest tests/test_finetune.py tests/test_student.py tests/test_data_math.py tests/test_early_stop.py` → 34 passed, 1 skipped(QAD CUDA). 10/27 18시 길이 정정(약 4 Hz 1 epoch)·GT 행 lm 끔·replay 제외 반영 뒤다.
  - 환경: `ALPAMAYO15`=1.5 release config.json 폴더. `HF_HUB_CACHE`는 4B 파일을 8B 이름으로도 가리키는 스크래치 캐시다. upstream `save_pretrained`가 기본 설정을 만들며 8B 프로세서를 찾는데, Mac에는 4B만 있다. 토크나이저·전처리는 같다.
  - 미세조정 테스트가 확인하는 것:
    - 약 4 Hz 행(2 Hz + 0.2/0.3 s 반반, 정확히 2배), 길이·비중, 중복 없음, 행이 모자란 새 소스는 다른 오프셋으로 채움, 기존 소스는 2 Hz 행만, 고정 seed
    - `exclude`(replay에서 재라벨 행 제외)
    - keep·재라벨(미래만 src_i)
    - 항목 표시: GT 행만 드롭아웃, replay 행만 vkd_ok
    - 행별 손실: GT 행만인 배치에는 kv·vkd·lm이 없고, 섞인 배치의 kv는 replay 행만
    - 드롭아웃: 이력 토큰만 바뀌고 0 변위 토큰과 같으며, 목표는 실제 이력
    - 선택 규칙 4경우, 설정(레시피 값), 정적 간격 계산
- `scripts/ft_tiny.sh` (합성 루트: 기존 4 + R2 7 에피소드; 소형 T*와 더 좁은 s2 학생; 12 step, 배치 2, 측정 4 step마다) 통과(10/27 18시, 정정판). 결과는 `logs/student/ft_tiny/`에 있다.
  - `MixtureDataset`: 새 약 4 Hz 행 68 = 2 Hz 36 × 약 2이고, 총 113 = 68 ÷ 0.6이다. 새 A·B·D는 각자 약 4 Hz 행을 정확히 1번 썼다. 기존 소스는 2 Hz 행에서 비중대로다. 합성 replay만 행이 적어 반복했다.
  - GT 행만 든 step에는 `vit`·`traj_ce`·`fm`만 있다. `kv`·`lm`·`vkd`는 replay 행이 든 step에만 기록된다.
  - `ft_select`는 step 0·4·8·12에서 쟀다. `ft_choice` = step 12(일반 ADE 4.072 ≤ 한계 4.157), `final/` 저장.
  - 이어서 `ft_measure history/static`이 그 final에서 돌았다.
