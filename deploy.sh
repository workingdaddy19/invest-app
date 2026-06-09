#!/bin/bash
# ============================================================
# INVEST 추론 API — 빌드 & 배포 스크립트
# 사용법: bash deploy.sh [옵션]
#   bash deploy.sh          # 빌드 + ECR 푸시 + K8s 배포 (전체)
#   bash deploy.sh --build  # 빌드 + ECR 푸시만
#   bash deploy.sh --deploy # K8s 배포만 (이미지 재빌드 없이)
#   bash deploy.sh --reload # 모델 핫 리로드만
# ============================================================
set -euo pipefail

# ── 설정 ─────────────────────────────────────────────────────
ECR_ACCOUNT="891376975666"
ECR_REGION="ap-northeast-2"
ECR_REPO="invest-app"
ECR_URI="${ECR_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}"

NAMESPACE="mlops"
DEPLOY_NAME="invest-inference"
TAG="${IMAGE_TAG:-latest}"          # 환경변수로 오버라이드 가능

MODE="${1:---all}"                  # 기본: 전체 실행

# ── 색상 출력 헬퍼 ───────────────────────────────────────────
info()    { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
success() { echo -e "\033[1;32m[OK]\033[0m    $*"; }
error()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
step()    { echo -e "\n\033[1;36m▶ $*\033[0m"; }

# ── 사전 확인 ────────────────────────────────────────────────
check_commands() {
    for cmd in docker aws kubectl; do
        command -v "$cmd" &>/dev/null || error "$cmd 명령을 찾을 수 없습니다."
    done
}

# ── Step 1: Docker 빌드 ───────────────────────────────────────
build_image() {
    step "Docker 이미지 빌드"
    docker build -t "${ECR_REPO}:${TAG}" .
    docker tag "${ECR_REPO}:${TAG}" "${ECR_URI}:${TAG}"
    success "빌드 완료: ${ECR_URI}:${TAG}"
}

# ── Step 2: ECR 로그인 & 푸시 ────────────────────────────────
push_image() {
    step "ECR 로그인"
    aws ecr get-login-password --region "${ECR_REGION}" \
        | docker login --username AWS --password-stdin "${ECR_URI}"
    success "ECR 로그인 완료"

    step "ECR 푸시: ${ECR_URI}:${TAG}"
    docker push "${ECR_URI}:${TAG}"
    success "ECR 푸시 완료"
}

# ── Step 3: K8s 배포 ─────────────────────────────────────────
deploy_k8s() {
    step "K8s 매니페스트 적용"

    # Namespace — 권한 없으면 경고 후 계속 (클러스터 관리자가 사전 생성 필요)
    if kubectl auth can-i create namespaces --all-namespaces &>/dev/null; then
        kubectl apply -f k8s/namespace.yaml
    else
        if kubectl get namespace "${NAMESPACE}" &>/dev/null; then
            info "Namespace '${NAMESPACE}' 이미 존재 — 건너뜀"
        else
            echo ""
            echo "  ⚠️  Namespace 생성 권한 없음."
            echo "  클러스터 관리자에게 아래 명령 요청:"
            echo "    kubectl create namespace ${NAMESPACE}"
            echo "    kubectl create secret generic mlflow-auth \\"
            echo "      --from-literal=username=<id> --from-literal=password=<pw> \\"
            echo "      -n ${NAMESPACE}"
            error "Namespace '${NAMESPACE}' 없음 — 관리자에게 생성 요청 후 재실행"
        fi
    fi

    # Secret 존재 여부 확인
    if ! kubectl get secret mlflow-auth -n "${NAMESPACE}" &>/dev/null; then
        error "mlflow-auth Secret이 없습니다.\n  kubectl create secret generic mlflow-auth \\\n    --from-literal=username=<id> --from-literal=password=<pw> \\\n    -n ${NAMESPACE}"
    fi

    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    success "매니페스트 적용 완료"

    step "롤아웃 대기 (최대 3분)"
    kubectl rollout status deployment/"${DEPLOY_NAME}" \
        -n "${NAMESPACE}" --timeout=180s
    success "배포 완료"

    step "Pod 상태 확인"
    kubectl get pods -n "${NAMESPACE}" -l app="${DEPLOY_NAME}"
}

# ── Step 4: 모델 핫 리로드 ───────────────────────────────────
reload_model() {
    step "모델 핫 리로드"
    POD=$(kubectl get pod -n "${NAMESPACE}" \
        -l app="${DEPLOY_NAME}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [[ -z "$POD" ]] && error "실행 중인 Pod가 없습니다."
    kubectl exec -n "${NAMESPACE}" "${POD}" \
        -- curl -s -X POST http://localhost:8080/model/reload | python3 -m json.tool
    success "모델 리로드 완료"
}

# ── 헬스 체크 ────────────────────────────────────────────────
health_check() {
    step "헬스 체크"
    POD=$(kubectl get pod -n "${NAMESPACE}" \
        -l app="${DEPLOY_NAME}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [[ -z "$POD" ]] && { info "Pod 없음 — 헬스 체크 건너뜀"; return; }
    kubectl exec -n "${NAMESPACE}" "${POD}" \
        -- curl -s http://localhost:8080/health | python3 -m json.tool
}

# ── 메인 ─────────────────────────────────────────────────────
main() {
    echo "========================================"
    echo "  INVEST 추론 API 배포 스크립트"
    echo "  ECR : ${ECR_URI}:${TAG}"
    echo "  NS  : ${NAMESPACE}"
    echo "  MODE: ${MODE}"
    echo "========================================"

    check_commands

    case "${MODE}" in
        --all)
            build_image
            push_image
            deploy_k8s
            health_check
            ;;
        --build)
            build_image
            push_image
            ;;
        --deploy)
            deploy_k8s
            health_check
            ;;
        --reload)
            reload_model
            ;;
        *)
            echo "사용법: bash deploy.sh [--all|--build|--deploy|--reload]"
            exit 1
            ;;
    esac

    echo ""
    success "모든 작업 완료!"
    echo ""
    echo "  테스트 UI : kubectl port-forward svc/${DEPLOY_NAME} 8080:8080 -n ${NAMESPACE}"
    echo "              → http://localhost:8080/ui"
}

main
