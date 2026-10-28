# RunPod 설정 가이드 (8-GPU 본학습용)

목표는 남은 과정 전부를 RunPod 8-GPU 대여 한 번으로 끝내는 것이다. 남은 과정은 교사 Stage 1-T/1-C/2, soft target, 학생 A/B, QAD다.
- vast 쪽 공통 사항(B2 원본, 비용 감시, 환경 묶음)은 [VAST_SETUP.md](VAST_SETUP.md)를 따른다.
- 2026-10-25 기준이다. 재고와 가격은 그날 07:45 KST에 RunPod 공개 GraphQL로 본 값이다.

## 0. 요약

| 항목 | 내용 |
|---|---|
| 지출 한도 | 기본 **시간당 $80**(계정 이력에 따라 자동 상향). 8× B200 $63.92/h, 8× H200 $42.32/h 모두 한도 안이다. 단, 배포하려면 잔액이 1시간 분량 이상 있어야 한다 |
| 크레딧 | **환불 불가**. 필요한 만큼 나눠 충전한다 |
| 8-GPU 재고 (10/25 07:45) | **8× B200 없음, 8× H100 SXM 없음.** 8× H200만 US-GA-2에 1곳($42.32/h, Low). 4× B200은 US-NC-2($31.96/h) |
| 데이터 | 네트워크 볼륨(데이터 센터 하나에 묶임, $0.07/GB/월)을 GPU 없이 미리 채운다. 단 US-GA-2는 네트워크 볼륨을 지원하지 않는다 |
| 이미지 | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` + B2의 sm100 venv tar. CUDA 12.8 이상 호스트만 고른다 |
| 비용 감시 | `tools/vast/idle_watchdog.sh`의 RunPod 모드(정지 = REST stop). 네트워크 볼륨의 `/workspace`는 남는다 |

재고 확인 (API 키 불필요, 수시로 다시 본다):
```bash
bash tools/vast/runpod_stock.sh 8
bash tools/vast/runpod_stock.sh 4 "NVIDIA B200"
```

## 1. 계정 (사용자)
1. https://www.runpod.io 에서 가입하고 이메일을 인증한다.
2. Settings에서 **2FA**를 켠다.
3. Billing에서 카드(Stripe)를 등록하고 크레딧을 충전한다.
   - 처음에는 소액이면 된다. 볼륨과 CPU pod 준비에는 $20 정도로 충분하다.
   - 8-GPU를 띄우기 직전에 하루치를 더 넣는다. 8× B200 기준 $64 × 24 h ≈ $1,530.
4. Billing에서 **잔액 부족 알림**을 켠다.
   - 잔액이 $0이 되면 네트워크 볼륨이 붙은 pod는 정지되고 데이터는 볼륨에 남는다.
   - 하지만 저장 요금을 못 내는 상태가 오래가면 볼륨이 삭제될 수 있다.

## 2. API 키, CLI, SSH 키
- **CLI**: PyPI의 `runpod` 1.12.0을 `uv tool install runpod`로 설치해 두었다(`~/.local/bin/runpod`).
  - 공식 `runpodctl`은 GitHub 바이너리라 설치하지 않았다. pod와 볼륨 조작은 REST API를 curl로 직접 부른다.
- **API 키 (사용자)**: https://console.runpod.io/user/credentials → API Keys → Create API Key. 이름은 `mac-cli`, 권한은 **All**로 한다.
  - Mac 터미널에서 아래를 실행해 저장한다. 화면과 셸 기록에 남지 않는다.
    ```bash
    mkdir -p ~/.runpod && (umask 077; read -rs "k?RunPod API key: " && printf '[default]\napi_key = "%s"\n' "$k" > ~/.runpod/config.toml); echo
    ```
  - 확인:
    ```bash
    runpod config --check && curl -s https://rest.runpod.io/v1/pods -H "Authorization: Bearer $(sed -n 's/^api_key = "\(.*\)"/\1/p' ~/.runpod/config.toml)" | head -c 200
    ```
- **SSH 공개키**: 공개키라 노출돼도 괜찮다.
  ```bash
  RP_KEY=$(sed -n 's/^api_key = "\(.*\)"/\1/p' ~/.runpod/config.toml)
  python3 -c 'import json,sys; print(json.dumps({"query":"mutation($k:String!){ updateUserSettings(input:{pubKey:$k}) { id } }","variables":{"k":sys.argv[1]}}))' "$(cat ~/.ssh/id_ed25519.pub)" |
    curl -s https://api.runpod.io/graphql -H 'content-type: application/json' -H "Authorization: Bearer $RP_KEY" -d @-
  ```
  - `runpod ssh add-key`(PyPI 1.12.0)는 "added"라고 출력하지만 실제로는 저장되지 않았다(10/25). 그래서 위 GraphQL로 넣었다.
  - 이 명령은 계정의 공개키 목록을 통째로 바꾼다.
  - 10/25 08:05 등록 완료. 계정 잔액 $30, `spendLimit` 80($/h)이다.
- **B2 읽기 전용 키 (사용자)**
  1. B2 → Application Keys에서 `runpod-read`를 만든다. 버킷은 `morai2026-alpamayo`만, 권한은 **Read Only**다.
  2. RunPod console → Secrets에 두 개를 만든다: `b2_key_id`(keyID), `b2_app_key`(applicationKey).
  3. pod 환경변수에 `{{ RUNPOD_SECRET_b2_key_id }}` 형식으로 넣으면, 값이 화면이나 명령에 나오지 않는다.

## 3. 저장소
- **네트워크 볼륨**
  - Secure Cloud 전용이고, 데이터 센터 하나에 묶인다.
  - pod를 만들 때만 붙일 수 있다. 붙이면 `/workspace`에 마운트된다.
  - 표준 속도는 200–400 MB/s다. 크기는 늘릴 수 있지만 줄일 수는 없다.
  - pod를 stop하든 terminate하든 데이터는 남는다.
- **크기는 1,500 GB**로 한다(교사 세션 10/25 추정: ZeRO-2 체크포인트가 단계마다 약 150 GB).
  - 요금: 처음 1 TB × $0.07 + 500 GB × $0.05 = 월 $95, 하루 약 $3.2.

| 용도 | 크기 |
|---|---|
| step0 데이터 | 203 GB |
| 체크포인트 | 29 GB |
| venv tar와 압축 푼 venv | 16 GB |
| HF 캐시 | 작음 |
| ZeRO-2 체크포인트 (약 150 GB, 단계마다 1개) | 약 450–600 GB |
| 단계별 최종 모델 (T* 1-T, 1-C, 2, 학생 A/B, QAD, 각 10–25 GB) | 약 100 GB |
| soft target과 여유 | 나머지 |

- **위치**: 8-GPU 재고가 보이는 데이터 센터를 고른다.
  - 네트워크 볼륨을 지원하고 CPU pod도 있는 미국 데이터 센터는 US-CA-2, US-NC-2(볼륨 지원은 확인 필요), US-NE-1, US-KS-2, US-MO-1 등이다.
  - **8× 재고가 실제로 뜬 곳에만 볼륨을 만든다.** 재고가 없는 곳에 만들면 그 볼륨을 쓸 GPU가 없다.
- **8-GPU 재고가 볼륨을 지원하지 않는 곳에만 있을 때** (지금의 US-GA-2 8× H200처럼): 볼륨 없이 띄우고, 컨테이너 디스크로 `b2 sync`한다.
  - 미국 안이면 200 GB에 15–30분이다. 그 시간 동안 8-GPU 요금이 나간다(8× H200 기준 $10–20).
  - 이때는 `networkVolumeId` 대신 `"volumeInGb":1500, "volumeMountPath":"/workspace"`를 쓴다. 이 pod 볼륨은 실행 중 월 $0.10/GB이고, pod를 terminate하면 같이 지워진다. 그래서 결과를 B2에 올린 뒤에만 terminate한다.

### 볼륨 만들기와 미리 채우기 (사용자 승인 후)
```bash
RP_KEY=$(sed -n 's/^api_key = "\(.*\)"/\1/p' ~/.runpod/config.toml)
curl -s -X POST https://rest.runpod.io/v1/networkvolumes -H "Authorization: Bearer $RP_KEY" -H 'content-type: application/json' \
  -d '{"name":"morai-main","size":1500,"dataCenterId":"<DC>"}'
```
- 같은 데이터 센터에 **CPU pod**(cpu3c, 8 vCPU)를 볼륨을 붙여 띄운다. 그 안에서 B2로부터 받는다. GPU 요금은 들지 않는다.
  ```bash
  curl -s -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RP_KEY" -H 'content-type: application/json' -d '{
    "name":"prefill","computeType":"CPU","cpuFlavorIds":["cpu3c"],"vcpuCount":8,"cloudType":"SECURE",
    "dataCenterIds":["<DC>"],"networkVolumeId":"<VOL_ID>","imageName":"ubuntu:24.04","containerDiskInGb":20,
    "env":{"B2_APPLICATION_KEY_ID":"{{ RUNPOD_SECRET_b2_key_id }}","B2_APPLICATION_KEY":"{{ RUNPOD_SECRET_b2_app_key }}"},
    "dockerStartCmd":["bash","-c","apt-get update -qq && apt-get install -y -qq python3-venv > /dev/null && python3 -m venv /opt/b2 && /opt/b2/bin/pip install -q b2 && cd /workspace && for d in step0 coc env; do /opt/b2/bin/b2 sync --threads 32 --no-progress b2://morai2026-alpamayo/$d/ /workspace/$d/; done > /workspace/prefill.log 2>&1; touch /workspace/PREFILL_DONE; sleep infinity"]}'
  ```
- 끝나면(`/workspace/PREFILL_DONE`) CPU pod를 **terminate**한다. 볼륨은 남는다.
- 체크포인트는 8-GPU pod에서 HF로 받는다. 미국 데이터 센터에서 몇 분이면 된다.

## 3.5 장비와 시간 (교사 세션 추정, 10/25 08:00)
- 설정: Stage 1은 재계산(GC) 끔, 배치 4. 1× B200 실측으로 정했다.
- 여유분 1.3배를 포함한 총시간 기준이다.

| 노드 | 총시간 | 끝 (월 10:00 시작) | 비용 |
|---|---|---|---|
| RunPod 8× H200 ($42.32/h, 10/25 재고 있음) | 52–57 h | 수 15:00–19:00 | $2,200–2,410 |
| RunPod 8× B200 ($63.92/h, 10/25 재고 없음) | ~43 h | 화 ~05:00 | ~$2,750 |
| vast 8× B200 ($83–86/h, 한도 미해결) | ~43 h | 화 ~05:00 | $3,600–3,700 |

- H200은 sm90이라 venv tar로 `env/a1_5_sft_venv.tar`를 쓴다. B200은 `env/a1_5_sft_venv_sm100.tar`다.
- 단계별 시간은 [train/README.md](train/README.md) "B200 최적화와 단계 C 시간"에 있다.
- 학생 단계 시간은 아직 실측 전이다.

## 4. 8-GPU pod
- **이미지**: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`.
  - B2 `env/`의 **sm100용 venv tar**(B200 시험에서 빌드)를 같은 경로에 푼다.
  - H200/H100으로 바뀌면 sm90 tar(`env/a1_5_sft_venv.tar`)를 쓴다.
- `allowedCudaVersions`는 `["12.8","12.9","13.0"]`으로 한다. torch 2.8 cu128이 드라이버 570 이상을 요구하기 때문이다.
- SSH: 커스텀 이미지라 sshd를 직접 띄운다. RunPod이 계정 공개키를 `$PUBLIC_KEY`로 넣어 준다. `ports`에 `22/tcp`를 연다.
```bash
curl -s -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RP_KEY" -H 'content-type: application/json' -d '{
  "name":"main8","computeType":"GPU","cloudType":"SECURE","gpuTypeIds":["NVIDIA B200"],"gpuCount":8,
  "dataCenterIds":["<DC>"],"networkVolumeId":"<VOL_ID>","allowedCudaVersions":["12.8","12.9","13.0"],
  "imageName":"nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04","containerDiskInGb":200,"ports":["22/tcp"],
  "env":{"DPH_USD":"63.92","B2_APPLICATION_KEY_ID":"{{ RUNPOD_SECRET_b2_key_id }}","B2_APPLICATION_KEY":"{{ RUNPOD_SECRET_b2_app_key }}","HF_TOKEN":"{{ RUNPOD_SECRET_hf_token }}"},
  "dockerStartCmd":["bash","-c","apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server > /dev/null; mkdir -p ~/.ssh; echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys; service ssh start; sleep infinity"]}'
```
- 접속 정보는 `GET /v1/pods/<POD_ID>`의 `publicIp`와 `portMappings["22"]`로 알 수 있다. GraphQL에서는 `pod.runtime.ports`의 `privatePort 22`, `isIpPublic`으로 본다.
- **10/25 실제 대여에서 겪은 것**
  - REST `dataCenterIds` enum에 `EUR-IS-5`가 없어서 지정할 수 없었다.
  - `allowedCudaVersions`를 걸면 재고가 있어도 "no instances"가 났다. 빼고 띄운 뒤 `nvidia-smi`로 드라이버를 확인한다(US-GA-2 H200은 595.91, CUDA 13.2).
  - secret은 pod를 만들 때만 env로 들어간다. ssh 세션에는 안 보여서 `/proc/1/environ`에서 읽는다.
  - pod 볼륨(`/workspace`)은 MooseFS 네트워크 FS라 chown이 안 된다. rsync는 `-rlt`로 한다.
  - 학습 결과를 올리려면 B2 **쓰기** 키가 따로 필요하다. `runs/` prefix로 제한한 `runpod-runs`를 `/workspace/b2.env`에 사용자가 직접 넣는다.
- 이후 절차: `tools/vast/bootstrap.sh`(venv tar 복원) → watchdog → 교사 세션의 실행 절차서.

## 5. 비용 감시와 종료
- watchdog RunPod 모드:
  ```bash
  DPH_USD=63.92 BUDGET_USD=<예산> nohup bash /workspace/idle_watchdog.sh > /dev/null 2>&1 &
  ```
  - pod 전용 키(`RUNPOD_API_KEY`)로 `POST /v1/pods/$RUNPOD_POD_ID/stop`을 부른다.
  - 유휴 15분, 멈춘 작업 60분, 예산 도달 시 정지한다. 가짜 curl로 유휴 정지 경로를 시험했다(10/25).
- **stop**: GPU를 놓아주고, 컨테이너 디스크는 지워진다. 네트워크 볼륨의 `/workspace`는 남는다.
  - 다시 켤 때 GPU가 없을 수 있다("zero GPUs"). 8-GPU는 특히 그렇다. 그래서 큰 작업 도중에는 멈추지 않게 실행 절차서가 쉬지 않고 이어져야 한다.
- **terminate**: pod를 지운다. 네트워크 볼륨은 남는다.
- 다 끝나면 순서대로 한다.
  1. 결과를 B2 `runs/`에 올리고 대조한다.
  2. pod를 terminate한다.
  3. 볼륨을 지운다(`DELETE /v1/networkvolumes/<ID>`).
