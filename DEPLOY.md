# INVEST 추론 API — 인프라 배포 가이드

> **대상**: ECR 등록 및 EKS Pod 배포 담당 인프라팀  
> **작성일**: 2026-06-05  
> **AWS Account**: 891376975666 / Region: ap-northeast-2

---

## 전달 파일 목록

### 배포에 필요한 파일 (루트)

| 파일 | 용도 |
|------|------|
| `inference.py` | 추론 API 서버 소스 (단일 파일) |
| `requirements.txt` | Python 패키지 목록 |
| `Dockerfile` | 컨테이너 빌드 명세 |
| `.env.example` | 환경변수 형식 참고용 |
| `deploy.sh` | **빌드 + 배포 자동화 스크립트** |

### K8s 매니페스트 (`k8s/`)

| 파일 | 용도 | 적용 순서 |
|------|------|----------|
| `namespace.yaml` | `invest-inference` 네임스페이스 생성 | 1 |
| `secret-template.yaml` | MLflow 인증 Secret 참고용 (직접 apply 금지) | 2 (명령으로 생성) |
| `deployment.yaml` | Pod 배포 명세 (replicas=2, HPA 연동) | 3 |
| `service.yaml` | ClusterIP 서비스 (port 8080) | 4 |
| `hpa.yaml` | 자동 스케일링 (min=2, max=10, CPU 60%) | 5 |

---

## 배포 절차 (deploy.sh 사용)

### 사전 준비

```bash
# 1. mlflow-auth Secret 생성 (최초 1회)
kubectl create namespace invest-inference

kubectl create secret generic mlflow-auth \
  --from-literal=username=admin \
  --from-literal=password=<실제패스워드> \
  -n invest-inference

# 2. ECR 리포지토리 생성 (최초 1회)
aws ecr create-repository \
  --repository-name invest-app \
  --region ap-northeast-2
```

### 전체 빌드 + 배포 (권장)

```bash
bash deploy.sh
# 또는
bash deploy.sh --all
```

### 단계별 실행

```bash
bash deploy.sh --build    # Docker 빌드 + ECR 푸시만
bash deploy.sh --deploy   # K8s 배포만 (이미지 재빌드 없음)
bash deploy.sh --reload   # 모델 핫 리로드만 (재배포 불필요)
```

### 특정 태그로 배포

```bash
IMAGE_TAG=v1.2.0 bash deploy.sh
```

---

## 수동 배포 절차

### Step 1 — Docker 빌드 및 ECR 푸시

```bash
ECR_URI=891376975666.dkr.ecr.ap-northeast-2.amazonaws.com/invest-app

aws ecr get-login-password --region ap-northeast-2 \
  | docker login --username AWS --password-stdin ${ECR_URI}

docker build -t invest-app:latest .
docker tag invest-app:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest
```

### Step 2 — K8s 매니페스트 배포

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Step 3 — 배포 확인

```bash
# Pod 상태 확인 (Running 2/2 되어야 정상)
kubectl get pods -n invest-inference

# 로그 확인 (모델 로드 완료 메시지)
kubectl logs -f deployment/invest-inference -n invest-inference

# 헬스 체크
kubectl exec -n invest-inference deployment/invest-inference \
  -- curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 4 — 테스트 UI 접속

```bash
kubectl port-forward svc/invest-inference 8080:8080 -n invest-inference
# 브라우저: http://localhost:8080/ui
```

---

## 배포 명세 요약

| 항목 | 값 |
|------|-----|
| ECR URI | `891376975666.dkr.ecr.ap-northeast-2.amazonaws.com/invest-app:latest` |
| Namespace | `invest-inference` |
| Replicas | 2 (초기) / 최대 10 (HPA) |
| CPU Request | 500m / Limit 2 core |
| Memory Request | 1Gi / Limit 4Gi |
| Port | 8080 (HTTP) |
| Health Probe | `GET /health` |
| 서비스 타입 | ClusterIP |
| 스케일 트리거 | CPU 사용률 60% 초과 |

---

## 환경변수 (deployment.yaml 내 설정)

| 변수명 | 값 | 출처 |
|--------|-----|------|
| `MLFLOW_TRACKING_URI` | `http://mlflow.mlflow.svc.cluster.local:80` | deployment.yaml |
| `MLFLOW_TRACKING_USERNAME` | (mlflow-auth Secret) | K8s Secret |
| `MLFLOW_TRACKING_PASSWORD` | (mlflow-auth Secret) | K8s Secret |
| `MODEL_CLS_NAME` | `invest-crel-classification` | deployment.yaml |
| `MODEL_REG_NAME` | `invest-crel-regression` | deployment.yaml |
| `MODEL_ALIAS` | `champion` | deployment.yaml |
| `S3_BUCKET` | `s3-an2-mlops` | deployment.yaml |
| `AWS_REGION` | `ap-northeast-2` | deployment.yaml |

---

## IRSA (S3 로그 쓰기 필요 시)

```bash
eksctl create iamserviceaccount \
  --cluster <cluster-name> \
  --namespace invest-inference \
  --name invest-inference-sa \
  --attach-policy-arn arn:aws:iam::891376975666:policy/S3WritePolicy \
  --approve
```

> S3 로그가 불필요하면 IRSA 없이도 추론 API 동작에는 문제 없습니다.

---

## 모델 교체 (재배포 불필요)

```bash
# MLflow UI에서 새 버전에 @champion alias 부여 후
bash deploy.sh --reload
```

---

## 문의

| 구분 | 담당 |
|------|------|
| 소스코드 / API | 개발팀 |
| MLflow 모델/Alias | ML 학습팀 |
| ECR / EKS / IRSA | 인프라팀 |
