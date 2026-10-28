# 출발 재라벨 확정본 (DATA_PLAN_R2.md 7절)

**상태: 확정.** 10/27 사용자가 샘플 13장을 보고 "진행"을 승인했다(메인 세션 경유). 이 폴더는 그 규칙 그대로 만든 확정본이다.

기존 수집 에피소드는 시작 직후 약 7.6 s(중앙값) 서 있다가 출발한다. 수집기의 재생 지연 8 s 때문이고, 그 결과 와우마요가 폐루프 출발 지점에 12–84 s 서 있었다. 이 폴더는 그 정지 구간 행의 정답 궤적을 "지금 바로 출발"로 바꾼 것이다. 원본(`_ego`, 라벨 parquet, `_index`)은 건드리지 않았다.

- 위치: winpc `D:/share/2026_MORAI_step0/_start_relabel/` = B2 `b2://morai2026-alpamayo/step0/_start_relabel/` (`step0/`는 D: 폴더의 미러다)
- 기준 인덱스: `_index/index.parquet` (= index_v3, `~/coc_runs/index_v3/index.parquet`와 같은 파일). 행 번호는 그 0.1 s τ 격자다.

## 규칙
1. **정지 구간**
   - 에피소드 시작부터 첫 GT 움직임 직전까지다. 첫 움직임은 v를 적분한 이동이 0.05 m를 넘는 때다(채점기 `time.move_m`).
   - 이력 1.5 s가 필요해서 i ≥ 15부터다.
   - 랩 출발(rv)과 체크포인트 출발(rs)을 같은 방식으로 다룬다.
2. **기준 행 `src_i`**
   - 출발 직전 속도가 0인 마지막 행이다(v ≤ 0.05 m/s).
   - 그래서 새 정답의 첫 점이 정지 이력과 이어진다(`v[src_i+1]` 중앙값 0.14 m/s).
3. **새 정답**
   - 행 i의 미래 64점을 `src_i`의 미래 64점으로 바꾼다. 시간만 `shift_s = τ_src − τ_i`만큼 앞당긴 것이다(중앙값 2.9 s, 90 % 5.5 s, 최대 9.6 s).
   - 형식은 `morai_vla.data.ego_trajectory`와 같다: 0.1 s 간격, `src_i`의 base_link 좌표계, z = 0, yaw만 회전.
   - 정지 중이라 행 i와 `src_i`의 자세 차이는 5 mm 미만이다.
   - 이력, 카메라, LiDAR, nav_text는 행 i 그대로다.
   - GT 위치와 비교한 끝점 오차는 경로 길이의 0.6 %다(중앙값, 200개).
4. **마스크**
   - `build_index.traj_ok_on_grid`(10/27 17:32 `j1 = max(j1, j0)` 수정본)를 `src_i`의 창 [τ_src, τ_src + 6.4 s]에 적용한다.
   - 속도 글리치(0.1 s당 1.5 m/s 초과)는 이력(i) 창과 미래(src_i) 창에서 본다.
   - 분할·셀·nav_text는 `build_index`와 같은 규칙이다. hold-out 에셋은 `splits.json` 목록을 쓴다. 기존 인덱스에도 있는 행으로 검사하니 셋 다 100 % 일치했다.
5. **신호 제외**
   - 검사 대상: 판정선이 정답 경로 길이 + 50 m 안에 있을 때만 신호를 본다. 수집 주행기가 판정선 20–40 m 앞 OSM 정지선에서 서므로, 그만큼 먼 신호도 정답에 영향을 준다. 예를 들어 CP3 출발 260건 중 162건은 출발 후 약 38 m 가서 다시 선다.
   - 판정선은 thresholds.yaml의 신호 교차로 4곳이다: 7294(s208.5), 6254(s315.9), 4188(s436.8), 4356(s609.9).
   - 비신호 정지선 7059(s72.6)와 6983(s160.0)은 검사하지 않는다.
   - 재라벨 조건: 행 i의 6.4 s 창과 `src_i`의 6.4 s 창에서 GT 신호가 같은 단일 상태로 이어져야 한다.
   - 조건을 못 맞추면 사유를 행 자신의 신호로 붙인다:

     | 행 i의 신호 | 사유 |
     |---|---|
     | 적 (창 안 GT 끊김 포함) | `signal_red` |
     | 황 | `signal_yellow` |
     | 그 밖이고 창에 GT가 빔 | `signal_unknown` |
     | 그 밖 | `signal_change` |

6. **장애물·이동 물체 제외** (winpc LiDAR BEV)
   - `obstacle_lidar`: 새 정답 경로 ±1.0 m 회랑 안에 지면 위 0.45–3 m 물체가 있다(2 m 구간에 셀 3개 이상).
   - `moving_object`: 행 i 스캔에는 있고 출발 순간 스캔 2장(3×3 팽창)에는 없는 물체가 경로 ±8 m 안에 있다(셀 4개 이상).
   - 정지 중에는 세계가 멈춰 있다. 예를 들어 CP11 옆 차로 트럭도 대기 내내 움직이지 않았다. 그래서 두 검사는 CP2 밖에서 1행(CP14)만 걸렸다.
   - CP2에서는 적신호 중 교차로(약 25 m)를 지나는 차량을 123행(회랑)과 416행(이동)에서 잡았다. 모두 신호로도 제외된다.
   - 양성 대조: 회랑을 옆으로 옮기면 정적 장애물 박스를 18 m에서 잡는다.
7. **그 밖의 제외**
   - `mask` / `glitch`: 기존 라벨 마스크(충돌, 체크포인트 누락, 차선 등)나 속도 글리치가 `src_i` 창에 걸린다.
   - `no_step0_lidar`: `lidar_index.npz`가 없다. MoraiDataset도 건너뛰는 5개 에피소드다.
   - 사유가 여럿이면 5 → 6 → 7 순서로 앞의 것 하나로 센다.

## 통계
- 전체 에피소드는 1,365개다.
  - 출발 전 정지 행이 있는 에피소드는 1,325개, 그 행은 104,233개다.
  - 나머지 40개는 이력 1.5 s를 채우기 전에 출발한다.
- **재라벨: 966개 에피소드, 54,285행 (1.508 h)**
  - 분할: train 51,118 / val 2,554 / val_asset 613
  - 54,285행 중 53,671행은 행 i가 기존 인덱스에도 있다. 기존 라벨은 "기다림"이다.
  - `src_i`는 전부 인덱스에 있다.
  - train에서 i와 src_i가 둘 다 인덱스에 있는 행은 50,555개다.

| 출발 | 재라벨 | 적 | 황 | 없음 | 바뀜 | 마스크 | 기타 |
|---|---|---|---|---|---|---|---|
| 랩 출발 | 5,219 | | | | | 450 | |
| CP1 (s72.6, 비신호 정지선) | 13,443 | | | | | 320 | |
| CP2 (판정선 19 m 앞) | 52 | 31,348 | | 306 | 14 | 10 | |
| CP3 (판정선 75 m 앞) | 3,099 | | 9,365 | 492 | 664 | 486 | |
| CP4 (판정선 87 m 앞, 출발 뒤 약 3.5 s까지 GT 없음) | | | | 1,291 | | | |
| CP5 | 436 | | | | | 175 | |
| CP6 (판정선 26 m 앞) | | | 59 | 3 | | | |
| CP7 | 533 | | | | | 255 | |
| CP8 (회전교차로 진입) | 25,454 | | | | | 1,384 | 글리치 68 |
| CP10 | 929 | | | | | 2,755 | |
| CP11 | 2,844 | | | | | 31 | |
| CP12 | 311 | | | | | 62 | |
| CP13 | 901 | | | | | 62 | |
| CP14 | 1,043 | | | | | 87 | 이동 물체 1 |
| 장면 정보 없음 | 21 | | | 33 | | | LiDAR 없음 227 |
| **합계** | **54,285** | 31,348 | 9,424 | 2,125 | 678 | 6,077 | 296 |

- 당긴 시간(`shift_s`)은 중앙값 2.9 s, 최대 9.6 s다.
- 정답 특성 (레시피 참고):
  - CP8 정답은 회전교차로 진입이라 느리다(6.4 s에 약 27 m). 재라벨의 47 %를 차지한다.
  - CP3 정답 중 적신호 상태가 같은 것은 출발 후 약 38 m에서 급정지(약 6 m/s²)한다. 기존 데이터 그대로다.

## 파일과 스키마

| 파일 | 행 | 열 |
|---|---|---|
| `start_relabel_index.parquet` | 54,285 (재라벨만) | `episode` (category), `i` (int64), `tau` (float64), `cell`, `split`, `nav_text`, `scene` (category), `fog` (bool), `hour` (int64): 여기까지 `_index/index.parquet`와 같다. 더한 열은 `src_i` (int64, 미래를 가져올 행), `src_tau` (float64), `shift_s` (float64) |
| `start_rows_all.parquet` | 104,233 (출발 전 전 행) | `episode`, `i`, `tau`, `src_i`, `shift_s`, `status` (위 사유 또는 `relabeled`), `signal_reason`, `line_id`, `line_d`(앞 판정선까지 m), `obstacle`, `s_obs`, `cells`, `moving_near`, `new_s`, `new`, `gone`, `mask_ok`, `glitch`, `in_index`(기존 인덱스에 있음), `split` |
| `start_future.parquet` | 1,325 (에피소드별) | `episode`, `start_cp` (0 = 랩 출발, −1 = 장면 정보 없음), `src_i`, `tau_dep`, `fut_len`, `fut_vmax`, `fut_xyz` (float32 192 = 64×3, 행 우선), `fut_yaw` (float32 64, `src_i` yaw 기준). `ego_trajectory(v, yaw, src_i)` 미래와 같고, 검증용이다 |
| `summary.json` | | 통계 |
| `figs/` | | 확인 그림 27장(무작위 재라벨 14 + 제외 사유별), 재라벨 프레임 모음 2장(200개 에피소드) |
| `code/` | | 생성 스크립트 사본 |

## 학습에서 쓰는 법
- 행 i의 카메라·LiDAR·nav_text·이력은 i에서 가져온다. 미래만 `ego_trajectory(ep.v, ep.yaw, src_i)`의 미래 두 개(xyz, rot)로 바꾼다.
- 미세조정 설정은 `train/configs/student_ft.yaml`의 `paths.start_relabel`이다(소스 `start_relabel` 5 %). 로더는 i와 `src_i`가 기존 인덱스에 있는 행만 쓴다.
- 같은 (episode, i)의 기존 "기다림" 라벨 53,671행은 인덱스에 그대로 있다.
  - replay 소스에서 이 행들을 빼야 같은 입력에 두 정답이 섞이지 않는다. `start_rows_all.parquet`에서 `status == "relabeled"`이고 `in_index`인 행이다.
  - 교사 soft target도 이 행들에서는 기다림이다.

## 생성 코드
- 원본: Mac 워크스페이스 `tools/start_relabel/`. 같은 사본이 이 폴더 `code/`에 있다.
  - `relabel_stage1.py` (Mac): 정지 구간, 새 정답, 마스크, 신호 검사, LiDAR 요청
  - `lidar_corridor.py`, `lidar_dynamic.py` (winpc Noetic chroot, nice 19 / ionice 3): 회랑 장애물, 이동 물체
  - `relabel_stage3.py` (Mac): 병합, 판정, 파일 쓰기
  - `relabel_figs.py`, `samples.py`, `pull.py`: 확인 그림, 사용자 샘플, winpc 프레임 추출
- 사용자 샘플 그림 13장: Mac 워크스페이스 `reports/start_relabel_samples/`, 사본은 이 폴더 `figs/samples/`

## 다시 돌리기
- 스크립트는 자기 폴더를 작업 폴더로 쓴다(`relabel/` 중간 파일, `start_relabel/` 결과). iCloud 밖(예: scratch)에 복사해서 돌린다.
- 순서:
  1. Mac: `uv run --with pandas,pyarrow,numpy,scipy,pyyaml python relabel_stage1.py` → `relabel/lidar_req.json`. `relabel/` 안에 winpc `_index/index.parquet`(→ `index_orig.parquet`)와 `splits.json`(→ `splits_orig.json`)을 먼저 복사해 둔다.
  2. winpc: `lidar_req.json`과 `lidar_corridor.py`, `lidar_dynamic.py`를 `~/start_relabel/`(step0 `run.sh` 사본과 같은 폴더)에 둔다. `nice -n 19 ionice -c3 ./run.sh lidar_corridor.py k 4`, `lidar_dynamic.py`도 같은 방식으로 k = 0..3.
  3. 결과 `lidar_out.*.jsonl`, `lidar_dyn.*.jsonl`을 `relabel/`로 가져와 Mac에서 `relabel_stage3.py`.
  4. 그림: `relabel_figs.py req`로 `pull_req.json`을 만든다. winpc에서 `pull.py`로 프레임을 뽑아 `pull_out/`로 가져온 뒤 `relabel_figs.py plot`.
- 결과는 winpc `D:/share/2026_MORAI_step0/_start_relabel/`에 있다.
