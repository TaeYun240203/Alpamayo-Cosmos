# vast.ai 설정 가이드 (교사 T\* 학습용)

[MODEL_PLAN.md](MODEL_PLAN.md) 4장의 2–5단계를 vast.ai에서 돌리기 위한 준비 절차다. 2026-10-24 기준이다.
- 가격과 매물은 그날 17:30 KST에 `vastai search offers`로 본 값이다. 시장 가격이라 수시로 바뀐다.
- 학습 코드와 단계 B 실행 절차는 교사 코드 세션([교사 T\* 학습 코드 작성](#local_fc2e91c8-e031-4dde-8485-3172a57c2953))이 맡는다. 이 문서는 계정, 인스턴스, 데이터, 체크포인트, 환경까지만 다룬다.

| 단계 | 장비 | 기간 | 예상 비용 |
|---|---|---|---|
| A | 로컬 | 지금 | 0 |
| B | 1× H100 80 GB, 디스크 500 GB | 1–2일 | **$90–200** |
| C | 8× H100 SXM (NVLink), 볼륨 1.5 TB | 약 60–85 h | **$1,200–1,700** (추정, 2.3절) |
| 원본 데이터 | Backblaze B2 버킷 1개 (~250 GB) | 계속 | 약 $1.7/월 |

**사용자만 할 수 있는 일**은 다음과 같다. 키와 토큰은 채팅에 붙여 넣지 말고 각자의 터미널이나 웹 화면에 직접 입력한다.
- 계정 생성
- 결제 수단 등록
- API 키·토큰 발급과 입력
- 클라우드 연결 등록


## 1. 계정과 CLI

### 1.1 계정 (사용자)
1. https://cloud.vast.ai 에서 가입하고, 받은 메일의 링크로 **이메일을 인증**한다. 인증 전에는 대여가 안 되고 지출 한도가 0에 가깝다.
2. Settings → Account Security에서 **2FA**를 켠다.
3. Billing → **Add Credit**으로 선불 충전한다. 최소 $5이고, 카드(Stripe)나 암호화폐로 낸다. 단계 B는 $200이면 충분하다.
4. 같은 화면에서 **Autobilling**을 켠다(카드만 가능).
   - 기준액은 하루 지출 정도로 잡는다. B는 $100, C는 $500.
   - Notification에서 저잔액 메일 알림을 기준액의 75%쯤으로 걸어 둔다.
   - 이유: 잔액이 0이 되면 인스턴스가 멈추고, 유예 기간이 지나면 **인스턴스·볼륨·데이터가 삭제**된다.
5. **지출 한도** (10/24 지원팀 답변 3회): 시간당 한도가 아니라 **계정이 쓸 수 있는 누적 총액**이다.
   - 계정 나이에 따라 **초 단위로** 오른다. 1일 $52, 7일 $340, 30일 $1,444이니 $4 + $48 × 계정 나이(일)이다. 시간당 약 $2씩 오른다.
   - 계정은 **2026-10-24 17:05 KST**에 만들었다.

| 누적 한도 | 시각 (KST) |
|---|---|
| 약 $17 | 10/24 23:35 |
| $52 | 10/25 17:05 |
| 단계 B 전체 약 $200 | 10/28 저녁 |
| B + C 약 $1,350–1,900 | 11/21–12/02 |

   - 시간당 비용이 $2보다 큰 장비는 한도에 막혀 쉬엄쉬엄 돌 수밖에 없다.
   - **단계 C는 한도를 올려 받지 않으면 11월 하순까지 불가능하다.** 지원팀에 상향을 요청 중이다.

### 1.2 CLI (설치 완료)
Mac에 PyPI의 `vastai` 1.9.1을 `uv tool install vastai`로 설치해 두었다(`~/.local/bin/vastai`).

### 1.3 API 키 (사용자)
1. https://cloud.vast.ai/manage-keys/ → API Keys → **+New**. 이름은 `mac-cli`, 권한은 기본값(전체)으로 하고 Create를 누른다. 키는 한 번만 보이니 바로 복사한다.
2. Mac 터미널에서 아래를 실행한다. 프롬프트가 뜨면 키를 붙여 넣고 Enter를 친다. 화면과 셸 기록에 남지 않는다.
   ```bash
   mkdir -p ~/.config/vastai && (umask 077; read -rs "k?vast API key: " && printf %s "$k" > ~/.config/vastai/vast_api_key); echo
   ```
3. 확인:
   ```bash
   vastai show user
   ```
   id, email, credit이 나오면 된다.
4. SSH 공개키 등록. 공개키라 노출돼도 괜찮다.
   ```bash
   vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
   ```


## 2. 인스턴스 선택

### 2.1 필터의 의미

| 필터 | 값 | 이유 |
|---|---|---|
| `gpu_name`, `gpu_ram` | H100_SXM (80 GB). B는 H100_NVL·H200도 허용 | C와 같은 sm90에서 메모리를 재야 C 예측이 맞는다 |
| `cuda_vers>=12.8` | 호스트 드라이버 ≥ 570 | recipe가 torch 2.8.0 cu128이다 |
| `reliability>0.98` | 가동률 기록 | 여러 날짜 학습이 도중에 끊기면 손해가 크다 |
| `verified=true` 또는 `datacenter=true` | 검증 기계 / Secure Cloud(Tier 2·3, ISO 27001) | Cloud Sync 자격 증명이 호스트에 잠시 복사되므로 공식 문서가 Secure Cloud를 권장한다 |
| `disk_space` | B ≥ 500, C ≥ 1500 GB | 생성 후에는 크기를 바꿀 수 없다 |
| `inet_down` | ≥ 1000 Mb/s 권장 | 203 GB를 1 Gb/s로 받으면 약 27분 걸린다 |
| `inet_down_cost` | 마우스를 올려 확인 | 호스트에 따라 $0–40/TB를 받는다. 203 GB에 $0–8 |
| `storage_cost` | 마우스를 올려 확인 | $0.13–1.00/GB/월. **멈춘 인스턴스도 과금**된다 |
| `duration` | B ≥ 3일, C ≥ 7일 | 최대 대여 기간 |
| `bw_nvlink` (C) | ≥ 400 GB/s (SXM NVSwitch) | ZeRO-2 그래디언트 all-reduce 때문이다. PCIe만 있는 8×는 피한다 |

### 2.2 단계 B: 1× H100

검색 명령(가격 순):
```bash
vastai search offers 'num_gpus=1 gpu_name in [H100_SXM,H100_NVL,H200,H200_NVL] reliability>0.98 disk_space>=500 inet_down>=1000 cuda_vers>=12.8 duration>=3' --storage 500 -o 'dph_total'
```

10/24 결과 (500 GB 디스크 포함 시간당 가격):

| 후보 | 위치 | $/h | 비고 |
|---|---|---|---|
| **H100 SXM** (Secure Cloud) | CZ | **4.17** | C와 같은 GPU다. 인터넷 8 Gb/s, 다운로드 $40/TB. 1순위 |
| H100 NVL 94 GB (Secure Cloud) | CZ | 3.65 | SXM보다 약간 느리다. 프로파일링할 때 80 GB로 메모리를 제한해서 잰다 |
| H200 NVL 141 GB | US-MT | 4.29 | 메모리가 남아 C 예측에는 덜 맞다 |

- **비용**: 24–48 h × $3.65–4.17 = **$88–200**에 다운로드 약 $8이 더해진다.
- **`--onstart-cmd`가 필요한 이유** (10/24 실측): `nvidia/cuda` 이미지에서는 sshd가 `bad ownership or modes for file /root/.ssh/authorized_keys`로 키를 거부했다(`vastai logs <ID>`로 확인). 실행 중인 인스턴스에서는 고칠 방법이 없다(`vastai execute`는 멈춘 인스턴스만 된다). 그래서 시작할 때 권한을 고친다.
- **고르는 기준: 시간당 가격만 보면 안 된다. 받는 비용을 더한 총비용으로 비교한다.**
  - 총비용 = 시간당 가격 × (데이터를 받는 시간 + 작업 시간) + 다운로드 요금.
  - 받는 시간은 B2(US West)까지의 거리로 정해진다. 표에 적힌 `inet_down`이 아니다.
  - 실측 (10/24–25)
    - 메릴랜드: 48–58 MB/s. 202 GB에 1 h.
    - 버지니아: 8 GB 단일 파일이 1분.
    - 남아공: 처음 30 MB/s였다가 **1.4 MB/s**로 떨어져 사실상 멈췄다. 그래서 7 GB만 받은 채 버렸다.
  - **미국(가능하면 서부) 호스트를 고른다.**
  - 디스크는 데이터 203 + 체크포인트 29 + venv 16 + 출력 = 약 260 GB면 된다. `disk_space>=300`으로 거르면 후보가 늘어난다.
- **대여 기록**

| 시각 (KST) | id | 장비 | 위치 | $/h | 결과 |
|---|---|---|---|---|---|
| 10/24 23:30 | 57525342 | H200 NVL | JP | 4.27 | SSH 키 권한 문제로 6분 뒤 반납 |
| 10/24 23:37 | 57525972 | H100 PCIe 80 GB | US-MD | 2.81 | 설치와 데이터 복사를 끝내고 00:54 정지. 그 사이 GPU를 다른 사람이 가져가서 다시 켜지 못하고 삭제 |
| 10/25 02:09 | 57541116 | H100 PCIe 80 GB | ZA | 3.34 | B2 복사가 1.4 MB/s로 떨어져 02:27 삭제 |
| 10/25 02:27 | 57543085 | H100 NVL 94 GB | US-VA | 3.11 | 단계 B 런북 전부 실행. 결과 `runs/stageB/`(82 files, 23.3 GB)를 B2에서 대조한 뒤 06:13 삭제 |

- **단계 B 결산 (10/25 06:13)**: 크레딧 $200 → $180.27, **총 $19.73**. 남은 인스턴스와 볼륨은 0개다.

| id | 쓴 시간 | 비용 |
|---|---|---|
| 버지니아 57543085 | GPU 3.8 h | 약 $11.7 |
| 메릴랜드 57525972 | GPU 1.3 h + 정지 1.5 h | 약 $5 |
| 남아공 57541116 | 0.3 h | 약 $1 |
| 일본 57525342 | 0.1 h | 약 $0.4 |

  - 나머지는 다운로드와 디스크 요금이다.

- **멈추면 GPU를 잃을 수 있다.** 단계 B 도중에는 stop보다 계속 돌리거나, 끝나면 결과를 B2에 올리고 destroy한다.
- **생성**:
  ```bash
  vastai create instance <OFFER_ID> --image nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 --disk 500 --ssh --direct --label stageB --onstart-cmd "bash -c 'for i in \$(seq 90); do chown root:root / /root /root/.ssh /root/.ssh/authorized_keys 2>/dev/null; chmod 755 /; chmod 700 /root /root/.ssh 2>/dev/null; chmod 600 /root/.ssh/authorized_keys 2>/dev/null; sleep 2; done' > /dev/null 2>&1 &"
  vastai show instances
  vastai attach ssh <INSTANCE_ID> "$(cat ~/.ssh/id_ed25519.pub)"   # 키가 안 붙었을 때만
  vastai ssh-url <INSTANCE_ID>
  ```
- **끝낼 때**: 결과를 B2에 올린 뒤(3.6절) `vastai destroy instance -y <ID>`. `stop`은 디스크 요금이 계속 나간다.

### 2.3 단계 C: 8× H100 SXM

검색 명령:
```bash
vastai search offers -n 'rentable=true num_gpus=8 gpu_name in [H100_SXM,H200,H200_NVL] bw_nvlink>=400 reliability>0.98 cuda_vers>=12.8 duration>=7' --storage 1500 -o 'dph_total'
```

10/24 결과. 매물이 적어 조건을 만족하는 H100은 1건뿐이었다.

| 후보 | 위치 | $/h | 비고 |
|---|---|---|---|
| **8× H100 SXM** (verified) | TW | **19.53** | NVLink 478 GB/s, RAM 1.5 TB, 96 vCPU, 디스크 $0.16/GB/월. 인터넷이 223 Mb/s로 느리다 → 3.5절의 볼륨 사전 적재로 GPU 대기 시간을 없앤다 |
| 8× H200 NVL (verified) | US-MS | 31.89 | 그날 오후 한때 있었다. 연산 속도는 H100 SXM과 비슷하다 |
| 8× B200 (Secure Cloud) | US | 83.95 | 비싸고 sm100이라 B 측정과 맞지 않는다 |

**시간 추정** (단계 B 프로파일링 결과로 바꿀 값이다)
- 가정
  - 샘플당 토큰 약 1,750(5.2절의 프리필 M과 같다).
  - 학습량은 6 × 파라미터 × 토큰 FLOPs, H100 BF16 MFU 40%로 잡았다.
  - 1 epoch는 69.5 h를 2 Hz로 자른 약 50만 샘플이다.
- 단계별 (8 GPU 기준)

| 단계 | 시간 |
|---|---|
| Stage 1-T + Stage 1-C (VLM, recipe Stage 1 두 번, 학습 파라미터 약 9.5B, ZeRO-2), 합계 2–3 epoch | 8–12 h |
| Stage 2 (expert) | 약 7 h |
| 4 soft target | 약 3 h |
| 5 학생 A·B (교사 동시 순전파 포함) | 약 22 h |
| 계 | 약 42 h |

- 디버깅, 관문 평가, 재실행 여유로 1.5–2배를 잡으면 **60–85 h × $19.5 ≈ $1,200–1,700**이다.
- 가격 선택지
  - interruptible(bid)은 더 싸지만 다른 사용자에게 밀려 중단될 수 있어 여러 날짜 학습에는 권하지 않는다.
  - reserved 할인은 몇 달 단위 선불이라 이번 규모에는 맞지 않는다.


### 2.4 비용 안전장치 (필수)
인스턴스가 놀면서 요금이 나가는 일을 막는다. 특히 사용자에게 질문하고 답을 기다리는 동안이 위험하다.
- 모든 인스턴스에서 [`tools/vast/idle_watchdog.sh`](tools/vast/idle_watchdog.sh)를 띄운다.
  ```bash
  BUDGET_USD=40 EXTRA_USD=<다운로드 비용> PRIOR_USD=<이미 쓴 금액> nohup bash idle_watchdog.sh > /dev/null 2>&1 &
  ```
- 이 스크립트는 인스턴스 안에서 인스턴스 전용 키(`CONTAINER_API_KEY`)로 `vastai stop instance`를 부른다. 정지 조건은 둘이다(유휴, 멈춘 작업).
  - **유휴 15분**: 다음 중 어느 것도 없을 때.
    (학습 프로세스에는 `train/scripts/*.py` 실행도 들어간다: `check_release.py`, `lidar_onoff_ce.py`, `time_*.py` 등. 10/25 단계 B에서 CPU만 쓰는 구간이 유휴로 잡혀서 추가했다.)
    - GPU 사용률 10% 초과
    - 학습·설치·다운로드 프로세스
    - `/workspace/keepalive` 파일이 최근 15분 안에 갱신됨
  - **멈춘 작업 60분**: 프로세스는 살아 있는데 GPU가 60분 동안 놀 때.
  - **예산은 정지 조건이 아니다**(사용자 10/27). 추정 지출이 `BUDGET_USD`와 "계정 한도 − 이미 쓴 금액 − $3" 중 작은 값에 닿으면 로그에 `WARN` 줄만 남긴다. 크레딧이 모자랄 것 같으면 상자를 맡은 세션이 디스코드 DM으로 사용자에게 알린다. 공급업체가 강제로 내린 경우가 아니면 크레딧 때문에 내리지 않는다.
- 가짜 `vastai`·`nvidia-smi`로 유휴 정지가 동작하는 것을 확인했다(10/24).
- **작업 규칙**
  - 사용자에게 물어볼 일이 생기면, 돌고 있는 작업이 없는 한 **먼저 인스턴스를 stop한다**.
  - 긴 작업은 nohup으로 띄운다. 작업이 끝나면 watchdog이 15분 뒤 멈춘다.
- stop하면 디스크는 남지만 저장 요금은 계속 나간다(500 GB에 시간당 $0.1–0.5).
- 멈춘 동안 그 GPU를 다른 사람이 빌려 가면 다시 시작이 안 될 수 있다. 그래서 결과물은 수시로 B2에 올리고, 단계가 끝나면 destroy한다.
- 다운로드 요금이 무료인 호스트를 고른다. 체코 호스트는 200 GB에 약 $8을 받는다.

## 3. 데이터 배치

### 3.1 vast.ai 저장소 (공식 문서 확인)
- **컨테이너 디스크**
  - 인스턴스와 수명이 같다. stop하면 남고, destroy하면 지워진다.
  - 크기는 생성할 때 고정된다. stop 중에도 과금된다.
- **볼륨**
  - **local volume만** 있다. 만든 물리 기계에 묶이고, 다른 기계로 옮기거나 붙일 수 없다.
  - 인스턴스를 destroy해도 남고, GB 단위로 따로 과금된다.
  - 인스턴스 없이 볼륨만 빌릴 수 있다(`vastai create volume`). `vastai copy`로 클라우드 → 볼륨, 볼륨 → 볼륨 복사가 된다.
- **Cloud Sync / Cloud Copy**
  - 지원: Google Drive, Amazon S3, **Backblaze**, Dropbox, Hugging Face(Storage Bucket).
  - **멈춘 인스턴스나 볼륨에도** 복사된다. 자격 증명은 호스트에 잠시 복사되므로 버킷 하나로 범위를 좁힌 키를 쓴다.
  - S3 연결은 AWS용이라 R2 같은 사용자 지정 endpoint 입력란이 문서에 없다.
- **수수료**: 같은 기계·같은 LAN이 아닌 인스턴스 간 복사에는 호스트별 대역폭 요금이 붙는다.

### 3.2 권장: 원본 1부를 Backblaze B2에 둔다

볼륨은 기계에 묶여 있다. 그래서 B의 1-GPU 기계에 올린 데이터는 C의 8-GPU 기계로 그대로 넘어가지 않는다. 원본은 기계 밖 객체 저장소에 한 부만 둔다.
- 집 → B2 업로드는 한 번만 하고, 그동안 GPU는 빌리지 않는다.
- B와 C는 각각 B2에서 받는다. vast 데이터센터 회선이라 빠르다.
- C는 GPU를 빌리기 전에 그 기계의 볼륨에 미리 채워 둔다(3.5절). 그래서 느린 회선(223 Mb/s)이어도 GPU 요금이 나가지 않는다.

| 선택지 | 비용 (~250 GB) | 장점 | 단점 |
|---|---|---|---|
| **Backblaze B2** (권장) | $6.95/TB/월 → 약 $1.7/월. 다운로드는 저장량의 3배/월까지 무료(약 750 GB), 넘으면 $0.01/GB | vast Cloud Copy가 직접 지원한다. 멈춘 인스턴스나 볼륨에 바로 채울 수 있다. CLI `b2`가 PyPI에 있다 | 미국·EU 리전뿐이라 한국에서 올릴 때 지연이 크다. 그래서 32스레드로 병렬 업로드한다 |
| Cloudflare R2 | $0.015/GB/월 → 약 $3.6/월, 다운로드 무료 | 인천 엣지에서 받아서 업로드가 빠르다 | vast Cloud Copy 연결이 없다. 인스턴스 안에서 직접 받아야 해서 C의 느린 회선이면 GPU 대기 약 2 h(≈ $37)가 생긴다 |
| Hugging Face 비공개 Storage Bucket | 무료 계정은 비공개 100 GB라 부족하다. PRO($9/월)는 1 TB | vast가 지원하고, HF 계정을 어차피 쓴다 | 유료 구독이 필요하다 |
| 집에서 `vastai copy local:` 직접 | 0 | 계정이 하나 줄어든다 | 업로드 약 1 h 동안 GPU 요금이 나가고, C에서 또 올려야 한다 |

### 3.3 측정한 업로드 속도 (10/24)
- **방법**: 무작위 200 MB를 Cloudflare 속도 측정 엔드포인트(인천 ICN)로 POST했다.

| 출발지 | 연결 | 속도 |
|---|---|---|
| winpc WSL | 단일 연결 2회 | 52.5, 54.9 MB/s |
| winpc WSL | 동시 4연결 | 합계 55.9 MB/s |
| Mac | 단일 연결 | 54.1 MB/s |

- 회선 상한은 **약 54 MB/s(≈ 430 Mb/s)**다. Mac과 winpc는 같은 회선을 쓴다. 그래서 데이터가 있는 winpc에서 바로 올린다.
- **B2 실측 (10/24, winpc → US West)**: TCP 연결 지연 약 170 ms.

| 시험 | 내용 | 속도 |
|---|---|---|
| 작은 파일 | `_ego` + `_index` (190 MB, 약 1,370개), 32스레드 | 1.4 MB/s (초당 약 10개) |
| 큰 파일 | 31 에피소드 (4.72 GB), 32스레드 | **23.8 MB/s** |
| 큰 파일 | `run1020h` 에피소드 전부 (6.92 GB), 64스레드 | **25.5 MB/s** |

  - 디스크 읽기는 병목이 아니다(DrvFs 165–325 MB/s). 태평양 구간이 국내 측정(54 MB/s)의 절반으로 묶인다.
  - 스레드를 늘려도 거의 빨라지지 않는다.
- **예상 시간**
  - 남은 약 191 GB ÷ 25 MB/s ≈ **2.1 h**다. 시험 업로드한 11.8 GB는 이미 올라가 있어서 건너뛴다.
  - `_coc`의 작은 파일 1.5만 개가 초당 약 10개씩 올라가므로 **2–2.5 h**로 본다.
- 업로드하는 동안 집 회선 업로드가 꽉 찬다. 수집은 10/23에 끝나서 걸리는 작업은 없다.

### 3.4 B2 준비 (사용자)
1. https://www.backblaze.com/sign-up/cloud-storage 에서 가입한다.
   - 리전은 **US West**(기본값)로 한다. 계정 단위로 고정된다.
   - 2FA를 켜고, 결제 카드를 등록한다(10 GB 초과분 과금용).
2. Buckets → **Create a Bucket**
   - 이름: 전역에서 유일해야 한다. 예: `morai2026-alpamayo-xxxx`. 아래에서는 `<BUCKET>`으로 쓴다.
   - Files in Bucket: **Private**. Object Lock: Off.
3. 버킷의 **Lifecycle Settings**를 "Keep only the last version of the file"로 둔다. 다시 올린 파일의 이전 판이 저장 요금을 먹지 않게 하기 위해서다.
4. Application Keys → **Add a New Application Key**를 두 번 만든다. 둘 다 "Allow access to Bucket(s)"는 `<BUCKET>`만, Type은 **Read and Write**로 한다. 키는 한 번만 보인다.
   - `winpc-upload`: winpc 업로드용
   - `vast-sync`: vast 연결용. 호스트에 복사되는 키라 따로 만들어 두면 이 키만 폐기할 수 있다.
5. winpc에서 인증한다. CLI는 PyPI의 `b2` 5.0.2를 `~/b2tool`에 설치해 두었다. 프롬프트에 keyID와 applicationKey를 직접 입력한다.
   ```bash
   ssh winpc-wsl
   ~/b2tool/bin/b2 account authorize
   clear
   ~/b2tool/bin/b2 bucket get <BUCKET>
   ```
   - `authorize`는 성공하면 applicationKey를 JSON에 그대로 출력한다. 그래서 바로 `clear`로 화면을 지운다.
   - 버킷을 하나로 제한한 키는 `bucket list`가 "restricted to buckets" 오류를 낸다. 정상이고, 확인은 `bucket get`으로 한다. `bucketType: allPrivate`이 나오면 된다.
6. vast 연결: https://cloud.vast.ai/settings/ → Cloud Connections → **Connect to Backblaze**. 이름 `b2`, 그리고 `vast-sync`의 keyID와 applicationKey를 넣는다. 연결 id는 이렇게 본다:
   ```bash
   vastai show connections
   ```

### 3.5 업로드와 배치 명령

**버킷 구조**
```
b2://<BUCKET>/step0/      ← D:\share\2026_MORAI_step0\ 에서 lidar.npz만 뺀 전체 (24,591 파일, 202.6 GB)
                            에피소드마다 cams.bin, cams_index.npz, lidar.bin, lidar_index.npz, meta.json + _ego, _index, _coc, _logs
b2://<BUCKET>/coc/        ← Mac ~/coc_runs/coc_v1_1 (v1.1이 나오면). v1은 이미 step0/_coc 안에 있다
b2://<BUCKET>/ckpt/       ← 단계 B에서 받고 변환한 체크포인트 (4.4절)
b2://<BUCKET>/runs/       ← 단계별 결과물
b2://<BUCKET>/env/        ← a1_5_sft_venv.tar (단계 B에서 빌드한 venv 7.97 GB; C에서 같은 경로에 풀면 flash-attn 빌드 생략)
```

**① 시험 업로드 (5 GB, 사용자 승인 후)**
```bash
ssh winpc-wsl 'cd /mnt/d/share/2026_MORAI_step0 && time ~/b2tool/bin/b2 sync --threads 32 --no-progress _ego b2://<BUCKET>/step0/_ego'
```
387 MB에 작은 파일 1,365개다. 이어서 큰 에피소드 몇 개를 같은 방식으로 올린다.

**② 본 업로드 (사용자 승인 후)**

ssh가 끊겨도 계속 돌고, 다시 실행하면 이미 올라간 파일은 건너뛴다.
- `lidar.npz`(12.2 GB)는 GT 자세로 만든 옛 형식이라 읽지도 올리지도 않는다. 로더는 `lidar.bin` + `lidar_index.npz`, `_index/`, `_ego/`를 읽는다.
- **LiDAR는 10/24 밤에 다시 만들었다.**
  - 움직임 보정과 스캔 2장 겹치기는 속도와 yaw rate 적분만 쓴다. 학습 때는 `_ego` GT의 v·yaw_rate를, 추론 때는 competition_status 속도와 IMU yaw rate를 넣는다.
  - 공용 모듈은 [`train/morai_vla/lidar_bev.py`](train/morai_vla/lidar_bev.py)이고, 바뀐 내용과 정렬 비교 결과는 [tools/step0/README.md](tools/step0/README.md)에 있다.
  - 새 파일은 `lidar_index.npz`에 `odom` 키가 있다. 다시 만든 뒤 아래 ② 명령을 한 번 더 돌려 바뀐 `lidar.bin`·`lidar_index.npz`(약 16 GB)만 올린다.
- `_index/`가 나중에 다시 만들어지면 같은 명령을 한 번 더 돌린다. 바뀐 파일만 올라간다.
```bash
ssh winpc-wsl 'setsid nohup ~/b2tool/bin/b2 sync --threads 32 --no-progress --exclude-regex ".*/lidar\.npz$" /mnt/d/share/2026_MORAI_step0/ b2://<BUCKET>/step0/ > ~/b2_step0_upload.log 2>&1 < /dev/null &'
```

**③ 확인**
```bash
ssh winpc-wsl '~/b2tool/bin/b2 ls -r b2://<BUCKET>/step0/ | wc -l'
ssh winpc-wsl '~/b2tool/bin/b2 sync --dry-run --threads 32 --exclude-regex ".*/lidar\.npz$" /mnt/d/share/2026_MORAI_step0/ b2://<BUCKET>/step0/ | head'
```
첫 명령은 winpc에서 `find /mnt/d/share/2026_MORAI_step0 -type f ! -name lidar.npz | wc -l`로 센 수와 같아야 한다(10/24 기준 24591). 둘째 명령은 아무것도 출력하지 않아야 한다(올릴 것 없음).

**④ CoC v1.1** (Mac → winpc → B2; 205 MB, 파일 31,426개, 10/24 올림)

macOS tar는 확장 속성을 `._파일명`(AppleDouble, 163 B) 파일로 같이 싸 보낸다. 그래서 `COPYFILE_DISABLE=1`을 붙인다. 10/24에 올린 `step0/_ego`(1,365개)와 `coc/coc_v1_1`에는 이 `._*` 파일이 섞여 있다. 로더는 파일 이름을 직접 열고 `glob('*.npz')`는 점 파일을 건너뛰므로 해는 없다. 그래서 B2 원본은 그대로 둔다.
```bash
COPYFILE_DISABLE=1 tar cf - -C ~/coc_runs coc_v1_1 | ssh winpc-wsl 'mkdir -p ~/coc_up && tar xf - -C ~/coc_up && ~/b2tool/bin/b2 sync ~/coc_up/coc_v1_1 b2://<BUCKET>/coc/coc_v1_1'
```

**⑤ 단계 B: 인스턴스로 가져오기**

Mac에서 실행한다. 호스트가 복사하므로 인스턴스에 키를 넣지 않는다.
```bash
vastai cloud copy --src /<BUCKET>/step0/ --dst /data/step0/ --instance <ID> --connection <CONN_ID> --transfer "Cloud To Instance"
vastai cloud copy --src /<BUCKET>/coc/ --dst /data/coc/ --instance <ID> --connection <CONN_ID> --transfer "Cloud To Instance"
```
처음에는 `_ego`처럼 작은 폴더로 경로 형식을 먼저 확인한다.

**데이터를 받는 방식 (10/25 단계 B에서 정함)**
- **컨테이너 디스크로 Cloud Copy를 하지 않는다.**
  - 호스트가 overlay 상위 계층에 직접 쓰기 때문에, 미리 조회된 경로는 복사가 끝나도 보이지 않는다(3.5절 아래 "Cloud Copy 실측"). 재부팅이 필요해진다.
  - 복사하는 동안 GPU 요금도 나간다.
- **1순위: 볼륨 미리 채우기** (아래 ⑥).
  - 볼륨은 overlay가 아니라 bind mount라서 위 문제가 없다.
  - GPU를 빌리기 전에 채우므로 GPU 요금이 들지 않는다.
- **2순위: 컨테이너 안에서 `b2 sync`**
  - 볼륨 매물이 없을 때 쓴다.
  - B2에서 이 버킷만 읽을 수 있는 **읽기 전용 키**(`vast-read`)를 사용자가 따로 만들어, vast Settings → Environment Variables에 `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`로 넣는다. b2 CLI는 이 두 변수를 자동으로 읽는다.
  - 인스턴스에서 실행한다.
    ```bash
    uv tool install b2 && b2 sync --threads 32 b2://<BUCKET>/step0/ /data/step0/
    ```
  - 이 경우 복사가 끝나기 전에는 학습 코드를 띄우지 않는다.

**⑥ 단계 C: GPU를 빌리기 전에 볼륨부터 채운다**
1. 8× 매물을 고르고 `machine_id`를 확인한다.
   ```bash
   vastai search offers ... --raw | jq '.[] | {id, machine_id, dph_total}'
   ```
2. 그 기계에 볼륨 매물을 찾아 볼륨을 만든다.
   ```bash
   vastai search volumes 'machine_id=<MID>'
   vastai create volume <VOL_OFFER_ID> -s 1500 -n morai_c
   vastai show volumes
   ```
3. B2 → 볼륨. GPU 요금이 나가지 않는다. 223 Mb/s면 250 GB에 약 2.5 h 걸린다.
   ```bash
   vastai copy b2.<CONN_ID>:/<BUCKET>/step0/ V.<VOL_ID>:/step0/
   vastai copy b2.<CONN_ID>:/<BUCKET>/ckpt/  V.<VOL_ID>:/ckpt/
   vastai copy b2.<CONN_ID>:/<BUCKET>/coc/   V.<VOL_ID>:/coc/
   ```
4. 다 차면 그 기계에 인스턴스를 만들고 볼륨을 /data에 붙인다.
   ```bash
   vastai create instance <OFFER_ID> --image nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 --disk 200 --ssh --direct --env '-v V.<VOL_ID>:/data' --label stageC
   ```
- **미검증**: 인스턴스가 붙지 않은 볼륨에 클라우드 → 볼륨 복사가 되는지는 아직 확인하지 못했다. 볼륨 문서에 `s3.<id>:/data/ V.<id>:/data/` 예시는 있지만, 지원팀은 data-movement 문서만 안내했다. C 전에 10 GB짜리 볼륨에 `_ego`를 복사해 시험한다. 안 되면 GPU를 빌린 뒤 받는다. 223 Mb/s 회선이면 약 2.5 h(≈ $50)가 든다.
- **위험**: 볼륨을 채우는 사이에 그 기계의 GPU가 다른 사람에게 대여될 수 있다. 그래도 볼륨 요금은 시간당 $0.3 정도라 손해가 작다. 다른 기계로는 `vastai copy V.<a>:/ V.<b>:/` 또는 B2에서 다시 받는다.

**Cloud Copy 실측 (10/24–25, 단계 B 인스턴스)**
- B2 US West → 메릴랜드 호스트: step0 202.6 GB, 파일 24,607개를 23:42–00:42에 받았다. 평균 48–58 MB/s다. 파일 수는 winpc와 같았다.
- **복사는 인스턴스당 한 번에 하나만 된다.** 다른 복사가 도는 중에 넣은 요청은 조용히 빠졌다. 그래서 하나가 끝난 것을 status에서 확인한 뒤 다음을 넣는다.
  ```bash
  vastai show instance <ID> --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['status_msg'][-120:])"
  ```
- **멈춘 인스턴스에도 된다.** CoC v1.1(파일 31,426개)을 정지 상태에서 받았다. 작은 파일이라 약 0.5 MB/s였다. 급하지 않은 복사는 GPU를 끈 채 돌린다.
- `vastai execute`(정지 상태에서 `ls`/`du`)는 CLI 1.9.1에서 응답 처리 오류가 나서 쓸 수 없었다.
- **복사 중에는 받는 경로를 건드리지 않는다** (10/25 버지니아에서 겪음).
  - Cloud Copy는 호스트가 컨테이너의 overlay 상위 계층에 직접 쓴다.
  - 그래서 아직 없던 경로를 컨테이너 안에서 미리 조회하면(로더의 존재 확인, `ls`, `stat` 등) 그 "없음"이 캐시된다. 이후 파일이 생겨도 `readdir`에는 이름이 보이지만 `stat`/`open`은 ENOENT를 낸다.
  - 실제로 rclone이 83 GiB를 썼을 때 컨테이너 안 `du`는 26.6 GB에서 멈춰 있었다.
  - 컨테이너 안에서는 `drop_caches`가 읽기 전용이라 캐시를 비울 수 없다. 복사가 끝난 뒤 `vastai reboot instance <ID>`로 컨테이너를 다시 띄우면 보인다. reboot 뒤에는 watchdog을 다시 띄운다.
  - 진행률은 컨테이너 안의 `du`가 아니라 `vastai show instance <ID> --raw`의 `status_msg`로 본다.

### 3.6 결과물 올리기 (destroy 전에 반드시)
```bash
vastai cloud copy --src /workspace/out/ --dst /<BUCKET>/runs/stageB/ --instance <ID> --connection <CONN_ID> --transfer "Instance To Cloud"
```


## 4. 체크포인트

### 4.1 게이트와 라이선스 (HF API, 10/24)

| 모델 | 게이트 | 라이선스 | 크기 | 고정 리비전 |
|---|---|---|---|---|
| `nvidia/Alpamayo-1.5-10B` | **없음** | 가중치 OpenMDW-1.1, 코드 Apache-2.0 | 22.2 GB | `7aba8293c09993f2e125c6819df05d7fa3e873ea` |
| `Qwen/Qwen3-VL-4B-Instruct` | **없음** | Apache-2.0 | 8.9 GB | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| `Qwen/Qwen3-VL-8B-Instruct` | 없음 | Apache-2.0 | 프로세서·토크나이저 파일만 받는다 | recipe의 `vlm_name_or_path`가 쓴다 |

- **수락할 화면이 없다.** 두 모델 모두 지금은 "Agree and access" 버튼이 없는 공개 모델이다. OpenMDW-1.1은 사용하면 동의하는 방식이다. 조건은 두 가지다.
  - 재배포할 때 라이선스 사본과 고지를 함께 넣는다.
  - 이 모델에 대한 특허·저작권 소송을 걸면 권리가 끝난다.
- **사용자가 판단할 것**: Alpamayo 1.5 모델 카드에는 "ready for non-commercial use. Commercial licensing available upon request"라는 문구가 있다. 라이선스 본문 자체에는 용도 제한이 없다. 대회 참가가 상업적 사용에 해당하는지는 사용자가 판단한다. MODEL_PLAN 6장의 출처·라이선스 목록에 이 표를 그대로 옮기면 된다.
- 참고: recipe의 `convert_release_config_to_training.py`가 가리키는 `nvidia/Cosmos-Reason2-8B`는 게이트(자동 승인)가 걸려 있다. 하지만 우리 경로(`convert_checkpoint.py to-a1`)는 그 모델을 쓰지 않는다.

### 4.2 HF 토큰 (선택, 권장)
공개 모델이라 토큰 없이도 받힌다. 다만 익명 요청은 속도 제한이 낮아서 읽기 전용 토큰을 하나 쓴다.
1. https://huggingface.co/settings/tokens → **Create new token**
2. Token type 탭에서 **Read**를 고르고, 이름은 `vast-read`로 한 뒤 **Create token**을 누른다. 값은 한 번만 보인다.
3. vast에 넣는다(채팅이나 저장소에는 넣지 않는다).
   - https://cloud.vast.ai/settings/ → **Environment Variables**
   - Key `HF_TOKEN`, Value에 토큰을 붙여 넣고 **+**, 그다음 **Save Edits**.
   - 이후 만드는 모든 인스턴스에 환경변수로 들어간다. 값은 암호화되어 저장된다.
   - 컨테이너 환경변수는 호스트가 볼 수 있다. 그래서 읽기 전용 토큰만 넣고, 프로젝트가 끝나면 HF에서 폐기한다.
   - 대안: 인스턴스에서 `hf auth login`을 실행한다. 입력이 화면에 보이지 않고 `$HF_HOME/token`에 저장된다.

### 4.3 받기와 A1 형식 변환 (인스턴스, 5장의 venv 안에서)
```bash
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
hf download nvidia/Alpamayo-1.5-10B --revision 7aba8293c09993f2e125c6819df05d7fa3e873ea --local-dir /workspace/ckpt/Alpamayo-1.5-10B
hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --local-dir /workspace/ckpt/Qwen3-VL-4B-Instruct
hf download Qwen/Qwen3-VL-8B-Instruct --include '*.json' --include '*.txt'
cd /workspace/alpamayo-recipes
python scripts/convert_checkpoint.py to-a1 --input /workspace/ckpt/Alpamayo-1.5-10B --output /workspace/ckpt/Alpamayo-1.5-10B-A1-format
```
- 셋째 줄은 프로세서 파일만 미리 받아 둔다.
- **교사 T\* 코드는 원본 폴더 `/workspace/ckpt/Alpamayo-1.5-10B`를 직접 읽는다.** A1 변환본은 recipe 기본 스크립트(`train_hf`)를 돌릴 때만 필요하다.
- recipe 기본 스크립트에서는 `model.checkpoint_path=/workspace/ckpt/Alpamayo-1.5-10B-A1-format`으로 넘긴다. 또는 `configs/models/ar1_5_base.yaml`의 `checkpoint_path`에 적는다.
- README의 `huggingface-cli download`는 같은 기능의 옛 이름이다.

### 4.4 C에서 다시 받지 않게
B에서 받고 변환한 것을 B2에 올린다. C는 3.5절 ⑥에서 볼륨으로 받는다.
```bash
vastai cloud copy --src /workspace/ckpt/ --dst /<BUCKET>/ckpt/ --instance <ID> --connection <CONN_ID> --transfer "Instance To Cloud"
```


## 5. 인스턴스 부트스트랩 (`recipes/alpamayo1_5_sft`, uv)

- **이미지**: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`(NVIDIA 공식). flash-attn을 PyPI 소스에서 빌드하므로 nvcc가 있는 devel 이미지여야 한다.
- **설치 출처**: 패키지는 PyPI, PyTorch, NVIDIA 공식 출처만 쓴다.
  - uv도 PyPI에서 설치한다.
  - recipe의 `alpamayo_r1`은 NVlabs GitHub에서 받는다.
  - flash-attn의 GitHub 사전 빌드 wheel 대신 `FLASH_ATTENTION_FORCE_BUILD`로 직접 빌드한다.
```bash
apt-get update && apt-get install -y --no-install-recommends git python3.12-venv python3.12-dev build-essential ninja-build tmux
python3 -m venv /opt/uvtool && /opt/uvtool/bin/pip install -q uv && ln -sf /opt/uvtool/bin/uv /usr/local/bin/uv
cd /workspace && git clone https://github.com/NVlabs/alpamayo-recipes.git && cd alpamayo-recipes && git checkout 2e907da
cd recipes/alpamayo1_5_sft
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=16
uv venv --python /usr/bin/python3.12 a1_5_sft && source a1_5_sft/bin/activate
uv sync --active
python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.get_device_name(), flash_attn.__version__)"
```
- `MAX_JOBS`는 nvcc 작업 하나에 RAM 4–8 GB가 드는 것에 맞춘 값이다.
- flash-attn 빌드에 torch가 아직 없다는 오류가 나면 두 번에 나눠 설치한다:
  ```bash
  uv sync --active --no-install-package flash-attn
  uv sync --active
  ```
- 실측 (10/24, H100 PCIe 호스트, 128 vCPU, MAX_JOBS=10): `uv sync`부터 flash-attn 빌드 끝까지 **약 24분**. RAM이 125 GB라 MAX_JOBS를 16 대신 10으로 낮췄다.
- 이 절의 명령은 [`tools/vast/bootstrap.sh`](tools/vast/bootstrap.sh)에 그대로 들어 있다. 비용 감시도 이 스크립트가 맨 먼저 띄운다.
- 학생 smoke (10/25, vast 57587913 H100 NVL, 09:49–11:27, 약 $5)에서 겪은 것:
  - `hf download --revision <commit>`은 HF 캐시에 `refs/main`을 만들지 않는다. 그래서 오프라인에서 repo id로 `from_pretrained`하면 실패한다. `get_ckpt.sh`가 이제 `refs/main`을 써 준다.
  - 학생 전체 크기 smoke의 체크포인트가 약 107 GB라 150 GB 디스크가 찼다. 학생용 상자는 **300 GB 이상**으로 빌린다.
  - winpc → Mac → 미국 tar 스트림은 5 MB/s 안팎이다(27 에피소드 6.8 GB에 27분). 상자를 다시 쓸 일이 있으면 시험용 subset을 B2에 올려 두고 받는 편이 빠르다.
- 체크포인트 받기는 [`tools/vast/get_ckpt.sh`](tools/vast/get_ckpt.sh)를 쓴다. 별도 venv라서 빌드와 동시에 돈다. 실측으로 Alpamayo 21 GB와 Qwen 4B 8.3 GB를 받는 데 몇 분 걸렸다.
  ```bash
  BUDGET_USD=40 EXTRA_USD=<다운로드 비용> PRIOR_USD=<이미 쓴 금액> MAX_JOBS=10 nohup bash /workspace/bootstrap.sh > /workspace/bootstrap.log 2>&1 < /dev/null &
  nohup bash /workspace/get_ckpt.sh > /workspace/get_ckpt.log 2>&1 < /dev/null &
  ```
- **C에서 빌드 시간을 아끼는 방법**: C도 같은 이미지·경로·Python이면 B의 venv를 그대로 옮겨 쓸 수 있다. B에서 `tar cf /workspace/out/a1_5_sft_venv.tar -C /workspace/alpamayo-recipes/recipes/alpamayo1_5_sft a1_5_sft`로 묶어 B2에 올리고, C에서 같은 위치에 푼다.
- **교사 T\* 코드와 단계 B 실행 절차**(스모크 테스트, 메모리·I/O 프로파일링, 2단계 실행)는 교사 코드 세션이 이 위에 얹는다. 이 문서와 맞춘 경로는 `/data/step0`, `/data/coc`, `/workspace/ckpt/*`, `/workspace/out`, `HF_HOME=/workspace/hf`다.
- **단계 B 런북**: [train/README.md — 단계 B 런북](train/README.md#단계-b-런북-vastai-1-h100-80-gb). 이 venv(`a1_5_sft`)를 그대로 쓰고, 코드는 `/workspace/train`에 둔다.


## 6. 끝낼 때
- **메트릭 보존 (사용자 10/27, 필수):** 인스턴스를 지우거나 멈춰도 메트릭·로그·검증 결과를 잃으면 안 된다.
  - 학습을 시작할 때 [`tools/vast/metrics_sync.sh`](tools/vast/metrics_sync.sh)를 `--loop 1800`으로 함께 띄운다. 실행 디렉터리의 로그, json, jsonl, csv, png, 설정을 30분마다 B2 `runs/<run>/`로 올린다. 가중치와 옵티마이저 상태는 빼고, 그건 단계별 업로드로 올린다.
  - 매 검증 직후와 단계가 끝날 때도 한 번 돌린다.
  - `idle_watchdog.sh`에도 `METRICS_SRC`와 `METRICS_DST`를 넘긴다. 그러면 자동으로 멈추기 직전에도 동기화한다.
  - 지울 때는 Mac에서 [`tools/vast/safe_destroy.sh`](tools/vast/safe_destroy.sh) `<ID> <상자 안 run 디렉터리> <b2:// 경로>`로만 지운다. 이 스크립트는 동기화가 성공해야 destroy한다.
  - 아무것도 돌리지 않은 상자(예: 네트워크 불량)만 `FORCE=1`을 쓸 수 있다.
- 인스턴스는 위 스크립트로, 볼륨은 `vastai delete volume <VOL_ID>`로 지운다. 결과물을 B2에 올린 것을 먼저 확인한다.
- B2의 `vast-sync` 키는 폐기하고, HF의 `vast-read` 토큰도 폐기한다.
- vast API 키는 쓰지 않을 때 Keys 화면에서 Reset하거나 지운다.
