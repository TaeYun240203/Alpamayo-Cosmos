# 와우마요 가중치 보간 런북 (2026-10-29)

상태: **코드만 준비했습니다. 상자는 빌리지 않았습니다.** 대여는 사용자가 승인한 뒤에 합니다(CLAUDE.md §1).
이 문서는 dynbal run(vast 58129242)이 실패할 때 쓸 1번 대안입니다. 3번 대안(R2 ft 자기증류)은 다른 세션이 준비합니다.

## 1. 무엇을 섞는가

| | 체크포인트 | B2 |
|---|---|---|
| A | R2 ft final (10/28 오전 납품본의 QAD 전 BF16, 폐루프 차선 위반 16–25회/바퀴) | `runs/r2ft/ft/final/` |
| B | dynbal ft final (A에서 시작, 합성 정지 + 근거리 돌출) | `runs/dynbal/ft/final/` (dynbal이 끝나야 생김) |

- 식: 모든 텐서에 θ = (1 − α)·A + α·B. 실수 텐서는 fp32에서 섞고 원래 dtype(bf16 등)으로 돌립니다.
- 정수·bool 텐서는 A와 B가 같아야 하고, 그대로 복사합니다.
- B가 A에서 시작했으므로 두 모델은 같은 부모에서 1,000 step(lr 1e-6) 갈라졌습니다. 그래서 둘 사이 직선 위에서도 손실이 크게 튀지 않을 것으로 기대합니다(추정, 아직 확인 안 함).
- **QAD 결과(납품본)끼리는 섞지 않습니다.** QAD는 모델마다 층별 양자화 형식이 달라서, ft final끼리 섞은 뒤 그 결과로 QAD를 합니다(6절).

## 2. 만든 파일

| 파일 | 하는 일 | 확인 |
|---|---|---|
| `train/scripts/weight_interp.py` | `--a --b --alpha --out`. 샤드 단위로 스트리밍(메모리에 출력 샤드 하나). 샤드 이름, 샤드별 키·모양·dtype이 같은지 assert. config·index·나머지 파일은 A의 것을 복사. 샤드 이름과 safetensors 메타데이터도 A와 같아서 `StudentVLA.build(init=…)`·`ft_measure.BF16(…)`가 A처럼 읽습니다. `<out>.partial`에 쓰고 끝나면 이름을 바꿉니다. 이미 있는 `<out>`은 덮어쓰지 않습니다. | 아래 테스트 |
| `train/tests/test_weight_interp.py` | 가짜 2샤드 체크포인트(bf16·fp32·int64): α=0이면 A와 비트까지 같음, α=1이면 B, α=0.5면 평균. 파일 구성·메타데이터는 A와 같음. 키 추가, 키 누락, 모양 다름, 정수 버퍼 다름은 모두 실패. | Mac CPU **5 passed** (`uv run --with torch,safetensors,pytest`) |
| `tools/vast/interp_eval.sh` | 5090 상자에서 α를 고르기 위한 측정(3–4절). GPU 수만큼 작업을 나눠 돌립니다. | `bash -n` 통과. Hydra 덮어쓰기는 Mac에서 `student_ft`에 합성해 8개 세트가 의도한 행 목록으로 풀리는 것을 확인. 요약 코드는 가짜 결과로 돌려 봄 |

설정 파일은 새로 만들지 않았습니다. 측정 세트는 `interp_eval.sh` 안의 Hydra 덮어쓰기로 정하고, QAD는 기존 `student_qad_dyn.yaml`을 그대로 씁니다(6절).

## 3. 측정 세트 (α 평가)

지표는 FtSelect와 같습니다. `scripts/gate_student.py +sets=[…]`가 `early_stop.expert_ade`를 FtSelect와 같은 `FtValSamples`로 부릅니다(6.4 s, K=6, LiDAR 켬, 행마다 seed 0, 한 행씩 prefill). 세트마다 ADE·FDE·|횡오차|·|종오차|와 행 수 n이 `sets.json`에 남습니다.

| 세트 | 행 | 정의 | 누구의 FtSelect |
|---|---:|---|---|
| general | 1,000 | R2 run이 잰 바로 그 행 (`assets/stop/general_val.parquet`) | A (student_ft) |
| dynamic | ≤300 | R2 A+B 보류 장면, stride 10, seed 1 (student_ft 그대로) | A |
| static | ≤300 | 옛 C2 보류 행, stride 10, seed 1 (student_ft 그대로) | A |
| curve | 약 354 | R2 run이 잰 곡선 행 (`curve_val` 300 + `curve_val_r2` 54) | A |
| start_relabel | ≤300 | 출발 재라벨 val 행 (qgate 기록과 같은 정의) | 기록용 |
| red_stop | 300 | R1 적신호 접근 150 + 정지 150 (`assets/stop/red_val.parquet`) | B (student_dyn) |
| dynamic_stop | 187 | R2 A+B 보류 장면의 합성 정지 라벨 (`_dyn_relabel/dyn_val.parquet`) | B |
| popout | ≤300 | 근거리 돌출 val 행 (`step0_popout/_dyn_relabel/mix/popout_val.parquet`, 녹화 에피소드의 5 %) | **없음**: 우리 정의 |

- general과 curve는 행 목록을 직접 줍니다. 그래서 상자에 R1 일부만 받아도 FtSelect와 같은 행입니다.
- static·dynamic·start_relabel은 무작위로 고릅니다. 고를 후보가 되는 에피소드는 모두 받습니다(스크립트가 색인에서 계산). 그래서 FtSelect와 같은 행이 나옵니다.
- popout: Mac의 `student_dyn.yaml`에는 돌출 val 세트가 없습니다. 그래서 dynbal의 FtSelect도 돌출 val을 재지 않았을 것으로 봅니다(상자 설정은 직접 보지 못했습니다). B2에 `popout_val.parquet`가 없으면 스크립트가 이 세트를 빼고 로그에 남깁니다.
- **재현 확인**: A와 B 모두 final = 마지막 측정입니다(`rel_worse: null`). 그래서 α=0의 general·dynamic·static·curve는 B2 `runs/r2ft/ft/ft_select.json`의 마지막 측정과, α=1의 red_stop·dynamic_stop은 `runs/dynbal/ft/ft_select.json`의 마지막 측정과 거의 같아야 합니다. 크게 다르면 측정 경로를 먼저 의심합니다.
- 늦은 정지의 오프라인 대리 지표는 정지 세트의 |종오차|(lon)입니다. 부호가 없어서 "늦게 섬"과 "일찍 섬"을 구분하지 못합니다. 그래서 폐루프 확인이 필요합니다(7절).

**이력 의존도** (`scripts/ft_probe.py --more a<α>=<ckpt> …`): 10/28·10/29 검사와 같은 156행, 이력은 실제 / 0 / 차로 모양 세 가지, seed 2개입니다. s2와 모든 α를 같은 상자에서 잽니다. 요약은 `hdep.py`의 표(0 이력 이동, 합성 이력 이동, 이력별 ADE·횡오차)입니다. 10/28 기준값은 0 이력 이동 s2 4.43 m, R2 ft step 1000 0.84 m입니다.

## 4. 절차 1: α 평가 (vast 5090)

### 4.1 대여 전 (하나라도 빠지면 대여 금지)
1. dynbal ft final이 B2에 다 있는지 셉니다. 샤드 4개 + index + config, 최소 6개입니다. dynbal이 아직 학습 중이면 대여하지 않습니다.
2. `~/.alpamayo-state/no_rent`가 없는지 봅니다.
3. 게이트를 돌리고, 세 클라우드 표를 사용자에게 보여 줍니다. 디스크는 300 GB 이상입니다(체크포인트 6개 약 91 GB + R1 일부 약 40–60 GB(추정) + step0_r2 16 GB + venv·HF).
```bash
bash tools/vast/rent_gate.sh --purpose "interp alpha eval" --target "5 alphas x 8 sets + 156-row history probe" \
  --end "INTERP_DONE + summary on B2 runs/interp/<RUN>, then safe_destroy on the user's OK" \
  --b2 b2://morai2026-alpamayo/runs/dynbal/ft/final/ --expect 6 --gpu "num_gpus=1 gpu_name=RTX_5090 disk_space>=300"
```
4. B2 키가 vast 계정 env-var에 있는지 봅니다(생성 뒤에 넣으면 그 상자에는 안 들어갑니다).

### 4.2 대여 직후 (Mac)
```bash
ID=<instance id>; RUN=5090_$ID; RENTED=$(date +%s)
bash tools/vast/ledger.sh add $ID <\$/h> "interp alpha eval" "5 alphas x 8 sets + probe" "INTERP_DONE, destroy on OK"
url=$(vastai ssh-url $ID); hp=${url#ssh://}; HP="${hp%:*} -p ${hp##*:}"
COPYFILE_DISABLE=1 tar czf - --exclude __pycache__ --exclude '*.bak-*' --exclude train/outputs --exclude train/logs train tools/vast \
  | ssh -o StrictHostKeyChecking=no $HP 'mkdir -p /workspace/probe && tar xzf - -C /workspace'
N="$HOME/Library/Application Support/Claude/side-session-notes/local_e4c51256-0d44-44d7-9043-37993b76a8ca/hprobe_inputs"
scp -P ${hp##*:} "$N"/{keep.parquet,synth.parquet,episodes.txt,rows.csv,hdep.py} ${hp%:*}:/workspace/probe/
ssh $HP "RUN=$RUN setsid nohup bash /workspace/tools/vast/interp_eval.sh > /workspace/interp.log 2>&1 < /dev/null &"
```
- 156행 입력의 원본은 다른 세션의 임시 폴더(`…/c81d4fd0-…/scratchpad/hprobe/`)에 있어서 지워질 수 있습니다. 그래서 공유 노트 폴더 `hprobe_inputs/`에 복사해 두었습니다. 스크립트는 이 입력을 결과와 함께 B2 `runs/interp/<RUN>/inputs/`에도 올립니다.
- 바꿀 수 있는 것(환경변수): `ALPHAS`(기본 `0 0.3 0.5 0.7 1`, 양 끝은 `0`·`1`로 씀), `R1_SETS`, `R2_SETS`, `PROBE_K`(2), `A_B2`, `B_B2`.

### 4.3 상자 안에서 일어나는 일
1. R1 색인을 받고, R1 val 세트 5개와 156행이 여는 에피소드만 고릅니다 → `r2ft_stage.sh` R1_EPISODES 모드로 스테이징합니다. 함께 받는 것: step0_r2 전체, s2 final, venv, HF, gpu_check 300 s.
2. 동시에 A·B와 돌출 val 에피소드(센서, `_ego`, `_popout_relabel`, `_dyn_relabel`)를 받습니다.
3. `dyn_stage.py`로 R2·돌출의 가상 에피소드를 펼치고 `dyn_load_check.py`(CPU)로 로더를 검사합니다. R2가 실패하면 중단합니다. 돌출이 실패하면 popout 세트만 뺍니다.
4. GPU 작업은 GPU마다 돌아가며 맡깁니다. 순서: R2 세트(α마다) → 이력 검사 → R1 세트(α마다). R2 세트는 R1을 받는 동안 돕니다. 그 사이 CPU에서 α 0.3·0.5·0.7 체크포인트를 차례로 만듭니다. α 0·1은 받은 A·B를 그대로 씁니다(테스트로 비트까지 같음을 확인).
5. `summary.json`·`summary.md`(α × 세트, ADE / |횡오차|), `hdep.md`를 만듭니다. B2 `runs/interp/<RUN>/`에 올리고 파일 수를 셉니다. 끝 표시는 `/workspace/INTERP_DONE` 또는 `INTERP_FAILED`입니다. 메트릭 루프(600 s)는 `runs/interp/metrics/<RUN>/`에도 계속 올립니다.

### 4.4 감시 (내가 직접, 2분 주기)
```bash
ssh $HP 'grep -E "rc=|gpu check" /workspace/stage.log; tail -4 /workspace/interp.log; ls /workspace/INTERP_* 2>/dev/null
         nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ls /workspace/out/stageR2/interp/a*/r*/sets.json 2>/dev/null | wc -l'
```
- 끝낼 조건: `INTERP_DONE`/`INTERP_FAILED`, `FAILED` 줄, 10분 넘게 새 줄 없음, GPU < 5 % 3번 연속(스테이징 중 제외).
- 보고 양식: `HH:MM | 단계(스테이징/R2 기록 a…/검사/R1 기록 a…) | $/h·지출·잔액·바닥까지 N분 | ETA`.
- 30분 안에 첫 R2 기록(`sets.json`)이 안 나오면 로그를 B2에 올리고, 사용자에게 보고한 뒤 지웁니다.

### 4.5 끝난 뒤
- 결과: B2 `runs/interp/<RUN>/summary.md`, `hdep.md`, `a<α>/{r1,r2}/sets.json`, `probe/probe_rows.jsonl`.
- 유휴 감시가 20분 뒤 상자를 **중지**합니다. 삭제는 사용자 답을 받은 뒤 `bash tools/vast/safe_destroy.sh $ID /workspace/out/stageR2 b2://morai2026-alpamayo/runs/interp/metrics/$RUN`로 하고, 이어서 `ledger.sh close`.

## 5. α 고르는 규칙 (사용자가 정함)

기호: 지표 m은 낮을수록 좋습니다. m_A·m_B는 α=0·1의 값, m_α는 그 α의 값입니다.
- 정지 유지율 S(α) = (m_A − m_α) / (m_A − m_B): dynbal이 얻은 개선 중 남은 비율. dynamic_stop·popout의 ADE와 |종오차|에 씁니다.
- 차선 비율 Lα = m_α / m_A: curve |횡오차|와 0 이력 이동에 씁니다.

| 후보 | 규칙 (숫자는 예시, 사용자가 정함) | 성격 |
|---|---|---|
| ① 정지 우선 | S ≥ 0.9(dynamic_stop, popout 둘 다)인 가장 작은 α | 늦은 정지 위험을 먼저 막음 |
| ② 차선 우선 | curve 횡오차 L ≤ 1.1, 0 이력 이동 L ≤ 1.2인 가장 큰 α | R2 ft 차선 추종을 먼저 지킴 |
| ③ 교집합 | ①과 ②를 모두 만족하는 α. 여럿이면 큰 쪽(정지 쪽). 없으면 보간은 탈락 → 3번 대안 또는 dynbal 그대로 | 둘 다 |

- 공통 하한: general ADE가 min(m_A, m_B)의 1.1배 이하, red_stop이 A보다 10 % 넘게 나빠지지 않음.
- B가 dynamic_stop·popout에서 A보다 낫지 않으면 섞을 이유가 없습니다. 그때는 보간을 하지 않습니다.
- α를 0.1 단위로 더 좁히려면 같은 상자에서 `ALPHAS="0.4 0.6"`처럼 다시 돌립니다. 이미 잰 α는 건너뜁니다.

## 6. 절차 2: 고른 α → QAD → 납품 → 관문 (8×H100)

QAD는 반드시 ft final끼리 섞은 체크포인트에서 시작합니다. 파이프라인에서는 그 체크포인트를 `$OUT/ft/final`에 두고 `ft`를 끝난 단계로 표시합니다. 그러면 `qad deliver qgate`만 돕니다(`QAD_FROM=ft`).

### 6.1 쓰는 설정 (새 설정 없음)
- `FT_CFG=student_dyn`: qgate의 행(base val 500)과 기록 세트(`red_stop dynamic_stop curve start_relabel`)를 정합니다.
- `QAD_CFG=student_qad_dyn`: dynbal 상자와 같은 QAD입니다. student_dyn 혼합, 4.75 bits AutoQuantize, 800 step, lr 1e-6, final = 가장 낮은 eval_fm.
- **결정 필요**: QAD 800 step(lr 1e-6, 모든 파트 1배)은 dyn 혼합에서 한 번 더 학습하는 셈입니다. ft는 ViT 0.1배, expert 0.3배였습니다. 그래서 보간 모델이 dynbal 쪽으로 다시 끌려갈 수 있습니다.
  - 대안 하나는 `QAD_CFG=student_qad_r2`(R2 ft 혼합)입니다. 이 경우 반대로 정지가 약해질 수 있습니다.
  - 어느 쪽이든 qgate 기록(입력 = 보간 ft vs QAD)으로 끌려간 양을 잽니다. 기본은 dynbal과 같은 `student_qad_dyn`입니다. 레시피를 바꾸는 것이라 사용자가 정합니다.

### 6.2 대여 전
- 4.1과 같습니다. `--purpose "interp QAD+deliver"`, `--target "QAD 800 step + qgate"`, `--end "deliver + gate.json on B2 runs/interp/<PICK>"`, `--gpu "num_gpus=8 gpu_name in [H100_SXM,H200]"`.
- Brev도 검색합니다(`brev search gpu --gpu-name H100`). 총비용으로 고릅니다.
- 학습 데이터 전체가 필요합니다. R1 step0 전체(약 190 GB), step0_r2(`_dyn_relabel/mix_ring` 포함), step0_popout 전체.

### 6.3 대여 직후 (Mac)
```bash
ID=<id>; RUN=h100_$ID; ALPHA=<고른 α>; PICK=a$ALPHA
# (코드 묶음 전송은 4.2와 같음. Brev: B2 키를 /workspace/b2.env에 손으로 넣음, 유휴 감시 없음 → 끝나면 손으로 멈춤)
ssh $HP "RUN=$RUN RUNB2=runs/interp/$PICK INIT_B2=runs/r2ft/ft/final setsid nohup bash /workspace/tools/vast/r2ft_stage.sh > /workspace/stage.log 2>&1 < /dev/null &
  RUNB2=runs/interp/$PICK WAIT_GO=1 QADMINI=0 FT_CFG=student_dyn QAD_CFG=student_qad_dyn QGATE_SETS='red_stop dynamic_stop curve start_relabel' \
  setsid nohup bash /workspace/tools/vast/r2ft_chain.sh > /workspace/chain.log 2>&1 < /dev/null &"
```
- `INIT_B2=runs/r2ft/ft/final`이면 `$OUT/s2/final` = A입니다. B만 따로 받으면 됩니다.
- 체인은 R1이 다 오면 `WAIT_GO`에서 기다립니다. `GO`가 생기면 즉시 `pipeline.sh STAGES="ft qad deliver qgate"`를 시작합니다. ft는 끝남 표시로 건너뜁니다.

### 6.4 R1을 받는 동안 상자에서 (venv·ckpt rc=0 뒤)
```bash
ALPHA=<고른 α>; PICK=a$ALPHA; B=b2://morai2026-alpamayo; OUT=/workspace/out/stageR2
source /workspace/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate; cd /workspace/train
mkdir -p /workspace/ck/B && b2 sync --threads 16 --no-progress $B/runs/dynbal/ft/final/ /workspace/ck/B/
python scripts/weight_interp.py --a $OUT/s2/final --b /workspace/ck/B --alpha $ALPHA --out $OUT/ft/final
b2 sync --threads 16 --no-progress $OUT/ft/final/ $B/runs/interp/$PICK/ft/final/
echo "local $(ls $OUT/ft/final | wc -l) / B2 $(b2 ls $B/runs/interp/$PICK/ft/final/ | wc -l)"      # 같아야 함
mkdir -p $OUT/done && date '+%F %T' > $OUT/done/ft                                            # pipeline: ft 건너뜀
mkdir -p /data/step0_popout && b2 sync --threads 32 --no-progress --exclude-regex '.*/lidar\.npz$' $B/step0_popout/ /data/step0_popout/
python scripts/popout_mix.py stage /data/step0_popout
python scripts/dyn_stage.py /data/step0_r2 && python scripts/dyn_stage.py /data/step0_popout     # step0_r2 rc=0 뒤
python scripts/dyn_load_check.py /data/step0_r2 && python scripts/dyn_load_check.py /data/step0_popout && touch /workspace/GO
```
- 보간은 결정적입니다(CPU fp32 lerp). 평가 상자에서 잰 α 체크포인트와 비트까지 같습니다. 그래서 평가 상자에서 15 GB를 올리지 않고 여기서 다시 만듭니다.
- `GO`는 로더 검사가 둘 다 통과한 뒤에만 만듭니다.

### 6.5 결과와 관문
- 납품: B2 `runs/interp/<PICK>/deliver/` (QAD 직후 바로 올림), 관문 결과는 `deliver/gate.json`입니다.
- 관문은 기존 규칙 그대로 `ratio_vs_s2` < 1.25입니다(QAD ADE / 보간 ft ADE, 같은 500행). 기록: `qgate/gate_{s2,qad}/sets_*`.
- 돌출 val은 qgate 기록에 없습니다(student_dyn 세트에 없음). QAD 출력에서 따로 잽니다.
```bash
python scripts/gate_student.py --config-name student_ft data.root=/data/step0 paths.s2=$OUT/s2/final paths.r2_root=/data/step0_r2 \
  "+callbacks.ft_select.sets.popout=[{root:'/data/step0_popout',split:[val,val_asset],index:'_dyn_relabel/dyn_index.parquet',keep:'/data/step0_popout/_dyn_relabel/mix/popout_val.parquet',n:300}]" \
  +student=$OUT/qad/final +out=$OUT/qgate/gate_qad/sets_popout "+sets=[popout]" +k=6
```
- 관문을 통과하고 B2 파일 수를 센 뒤, 사용자 답을 받고 상자를 지웁니다(Brev는 손으로 멈춤·삭제).

## 7. 폐루프 확인 (필수)
- 오프라인 지표만으로 고르지 않습니다. 납품본을 폐루프 담당 세션에 넘깁니다(5080 엔진 변환 → 한 바퀴 + 동적 장애물 장면).
- 볼 것:
  - 차선 위반 회수/바퀴: R2 ft 16–25, r2stop 38–54
  - 신호 위반
  - 돌출 보행자 앞에서 서는지(멈춘 거리, 충돌 여부)
- 정지가 반만 남으면 늦게 섭니다. 그러면 충돌입니다. 오프라인 |종오차|가 B에 가까워도 폐루프에서 한 번이라도 박으면 탈락입니다(판정은 사용자).

## 8. 시간·비용 (추정)

| 단계 | 상자 | 시간 | 비용 |
|---|---|---|---|
| α 평가, 세트 전부 | vast 5090 1장 ($0.7–0.8/h) | 스테이징 약 25분 + GPU 2.5–4 h → **약 3–4.5 h** | $2–3.5 |
| α 평가, 세트 전부 | vast 5090 4장 (약 $3/h, 재고 있을 때) | 약 1.2–1.6 h | $3.5–5 |
| α 평가, R1은 general·curve만 | 5090 1장 | 약 2–3 h | $1.5–2.5 |
| 보간 + QAD + 납품 + 관문 | 8×H100 (Brev 약 $25/h) | 스테이징 30–40분 + QAD·관문 약 2.5 h → 약 3–3.2 h | $75–80 |

- 근거: 5090 차선 시험에서 추론 1번이 0.3 s였습니다(198행 × 3조건, 3분). FtSelect는 8×H200에서 측정 1번(약 2,000행)에 2–3분이었습니다. 이로부터 K=6 한 행을 0.4–0.8 s로 잡았습니다.
- 그러면 α 하나에 약 3,000행이고 5개면 1.7–3.4 h입니다. 이력 검사는 6모델 × 156행 × 3이력 × 2 seed = 5,616회 × 0.3 s ≈ 0.5 h, 모델 적재는 16번 × 약 1분입니다.
- 처음 들은 "5090 1장 약 1 h"는 세트를 다 재면 맞지 않습니다. 1시간 안에 끝내려면 GPU 4장을 쓰거나, 세트·α를 줄여야 합니다.
- QAD 2.5 h는 전달받은 추정값을 그대로 썼습니다. 직접 잰 값이 아닙니다.

## 9. 위험
- **늦은 정지 → 충돌**: 보간은 정지를 "반쯤" 남길 수 있습니다. 그래서 돌출 val과 폐루프 둘 다 필수입니다(7절).
- **QAD가 보간점을 옮김**: 6.1에 적은 대로 QAD 혼합에 따라 dynbal 쪽이나 R2 ft 쪽으로 다시 끌립니다. qgate 기록으로 잽니다.
- **선형 연결성 가정**: 중간 α에서 두 끝보다 둘 다 나쁠 수 있습니다(손실 장벽). 측정에 α=0·1을 함께 넣은 이유입니다.
- **돌출 val 정의**: 우리가 정한 것입니다(n 300, seed 1, 녹화 에피소드 5 %). dynbal 학습 중 FtSelect 값과 비교할 수 없습니다.
- **B 준비**: dynbal ft final이 B2에 다 있어야 합니다. dynbal이 중단 스위치로 멈추면 그 final은 중단 시점입니다(`ft_stop.json` 확인).
- **디스크**: 평가 상자는 300 GB 이상이어야 합니다. 스크립트는 220 GB보다 적으면 바로 멈춥니다.

## 10. 확인한 것 / 하지 않은 것
- 확인한 것:
  - `tests/test_weight_interp.py` 5 passed (Mac CPU, 가짜 텐서)
  - `bash -n tools/vast/interp_eval.sh` 통과
  - 스크립트의 Hydra 덮어쓰기를 `student_ft`에 합성했을 때 8개 세트가 위 표대로 풀림
  - 요약 표 코드를 가짜 결과로 실행
- 하지 않은 것:
  - 실제 15 GB 체크포인트 보간
  - 출력물의 `StudentVLA.build` 적재: 상자의 첫 기록 작업이 strict 적재로 확인합니다
  - 실제 색인으로 R1 에피소드 고르기, B2 `step0_popout` 구조 확인(문서로만 확인)
  - 시간 실측
- 다른 세션 소유 상자(58129242 dynbal, 58143395 이력 검사, 그 밖의 5090)는 건드리지 않았습니다. 커밋하지 않았습니다.
