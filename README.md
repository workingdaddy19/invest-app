# INVEST 투자적격 심사 추론 API

부동산담보대출 투자적격 심사 AI 모델(투자적격판단 + 적정금리평가)을 FastAPI로 서빙하는 EKS Pod 애플리케이션.

## 파일 구조

```
.
├── inference.py                   # 앱 전체 (FastAPI + 모델 로드 + 전처리 + 엔드포인트)
├── requirements.txt               # 패키지 의존성
├── .env.example                   # 환경변수 형식 참고용 (커밋 가능)
├── Dockerfile                     # 컨테이너 빌드
├── test_inference.py              # 단위 테스트 (Mock 기반, 로컬 실행 가능)
├── k8s/
│   ├── deployment.yaml            # Deployment (ECR: 891376975666, replicas=2)
│   ├── service.yaml               # ClusterIP Service (port 8080)
│   └── hpa.yaml                   # HPA (min=2, max=10, CPU 60%)
└── docs/
    ├── 01.plan-api                 # 설계 원문
    ├── inference.py                # 소스 원본 참고
    ├── inference_requirements.txt
    ├── test_inference_integration.py  # MLFlow 실 연결 통합 테스트
    └── invest_crel_model_2.ipynb   # 학습 노트북
```

---

## Windows 로컬 테스트 (담당부서 전달 전 검증)

### 1단계: Python 환경 준비

```powershell
# Python 3.11 설치 확인
python --version

# 가상환경 생성 (권장)
python -m venv .venv
.venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2단계: 단위 테스트 실행 (MLFlow/AWS 서버 불필요)

```powershell
# Mock 기반 테스트 - 외부 서버 연결 없이 로컬에서 바로 실행
pytest test_inference.py -v
```

**예상 출력:**
```
test_inference.py::TestHealth::test_ok                         PASSED
test_inference.py::TestHealth::test_model_not_loaded           PASSED
test_inference.py::TestModelInfo::test_returns_model_names     PASSED
test_inference.py::TestModelInfo::test_loaded_at               PASSED
test_inference.py::TestPredict::test_full_response             PASSED
test_inference.py::TestPredict::test_invest_yn_y               PASSED
test_inference.py::TestPredict::test_fair_rate_value           PASSED
test_inference.py::TestPredict::test_minimal_request           PASSED
test_inference.py::TestPredict::test_missing_required_field_422 PASSED
test_inference.py::TestPredict::test_model_not_loaded_503      PASSED
test_inference.py::TestPredict::test_invest_yn_n_when_prob_low PASSED
test_inference.py::TestModelReload::test_reload_success        PASSED
test_inference.py::TestPreprocess::test_output_shape           PASSED
...
```

### 3단계: FastAPI 서버 로컬 실행 (선택)

.env 파일이 있으면 실제 MLFlow 모델을 로드하여 서버 실행 가능.

```powershell
# .env 파일 생성 (MLFlow 접속 정보 필요)
copy .env.example .env
# .env 파일 열어서 MLFLOW_TRACKING_URI 등 실제 값 입력

# 서버 실행
python -m uvicorn inference:app --host 0.0.0.0 --port 8080 --reload
```

**로컬 API 테스트 (서버 실행 후):**

```powershell
# 상태 확인
Invoke-RestMethod -Uri http://localhost:8080/health

# 추론 테스트
$body = @{
    gpt_ivt_jg_seq = "JG_LOCAL_TEST_001"
    ltv_rte = 65.0
    dbt_rpy_coef_rte = 1.35
    ln_pd = 24.0
    bs_itt = 3.25
    gpt_ivt_kd_cd = "오피스빌딩"
    gpt_ivt_ser_dv_cd = "오피스"
    gpt_ivt_tp_cd = "선순위"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8080/predict -Method POST `
    -ContentType "application/json" -Body $body
```

### 4단계: MLFlow 실 연결 통합 테스트 (선택, EKS 클러스터 접근 필요)

```powershell
# kubectl port-forward로 MLFlow 터널링
kubectl port-forward svc/mlflow 5000:80 -n mlflow

# .env의 MLFLOW_TRACKING_URI를 로컬로 변경
# MLFLOW_TRACKING_URI=http://localhost:5000

# 통합 테스트 실행
python docs/test_inference_integration.py
```

---

## EKS 배포 (담당부서)

### 사전 준비

```bash
# Namespace 생성
kubectl create namespace invest-inference

# MLFlow 인증 Secret 생성 (실제 패스워드 입력)
kubectl create secret generic mlflow-auth \
  --from-literal=username=admin \
  --from-literal=password=<실제패스워드> \
  -n invest-inference
```

### Docker 빌드 및 ECR 푸시

```bash
# ECR 정보
ECR_URI=891376975666.dkr.ecr.ap-northeast-2.amazonaws.com/invest-inference

# ECR 리포지토리 생성 (최초 1회)
aws ecr create-repository --repository-name invest-inference --region ap-northeast-2

# 빌드
docker build -t invest-inference:latest .

# ECR 로그인 및 푸시
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin ${ECR_URI}

docker tag invest-inference:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest
```

### 배포

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# 배포 확인
kubectl rollout status deployment/invest-inference -n invest-inference
kubectl get pods -n invest-inference

# 로그 확인
kubectl logs -f deployment/invest-inference -n invest-inference
```

---

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버/모델 상태 확인 (K8s readinessProbe) |
| GET | `/model/info` | 현재 로드 모델 버전 정보 |
| POST | `/model/reload` | champion alias 변경 후 핫 리로드 (재배포 불필요) |
| POST | `/predict` | 투자적격 심사 추론 |

### POST /predict 예시

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gpt_ivt_jg_seq": "JG20260604001",
    "ltv_rte": 65.0,
    "dbt_rpy_coef_rte": 1.35,
    "ln_pd": 24.0,
    "bs_itt": 3.25,
    "gpt_ivt_kd_cd": "오피스빌딩",
    "gpt_ivt_ser_dv_cd": "오피스",
    "gpt_ivt_tp_cd": "선순위"
  }'
```

**응답:**
```json
{
  "request_id": "20260604100000123456",
  "gpt_ivt_jg_seq": "JG20260604001",
  "invest_yn": "Y",
  "invest_prob": 0.7821,
  "fair_rate": 5.4312,
  "cls_model": "invest-crel-classification@champion",
  "reg_model": "invest-crel-regression@champion",
  "inferred_at": "2026-06-04 10:00:00"
}
```

---

## 모델 교체 (재배포 불필요)

```bash
# 1. MLFlow UI에서 새 버전에 @champion alias 부여
# 2. /model/reload 호출 (클러스터 내부)
kubectl exec -n invest-inference deployment/invest-inference -- \
  curl -s -X POST http://localhost:8080/model/reload
```

---

## MLFlow Artifact 사전 준비 (모델 학습팀 협의)

| Artifact 경로 | 내용 | 비고 |
|--------------|------|------|
| `encoders/le_dict.pkl` | 범주형 컬럼 LabelEncoder | 없으면 hash 폴백 |
| `encoders/le_target.pkl` | 타깃 LabelEncoder (Y/N 복원) | 없으면 확률 0.5 기준 |

---

## 환경변수 목록

| 변수명 | 필수 | 기본값 | 설명 |
|--------|------|--------|------|
| `MLFLOW_TRACKING_URI` | Yes | - | MLflow 서버 주소 |
| `MLFLOW_TRACKING_USERNAME` | Yes | - | MLflow 인증 ID |
| `MLFLOW_TRACKING_PASSWORD` | Yes | - | MLflow 인증 PW |
| `MODEL_CLS_NAME` | No | `invest-crel-classification` | 분류 모델명 |
| `MODEL_REG_NAME` | No | `invest-crel-regression` | 회귀 모델명 |
| `MODEL_ALIAS` | No | `champion` | 모델 alias |
| `S3_BUCKET` | No | `s3-an2-mlops` | 추론 로그 S3 버킷 |
| `AWS_REGION` | No | `ap-northeast-2` | AWS 리전 |
