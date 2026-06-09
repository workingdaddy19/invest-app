"""
test_inference.py - 단위 테스트 (Mock 기반, MLFlow/AWS 서버 불필요)

실행:
    pip install -r requirements.txt
    pytest test_inference.py -v

참고:
    실제 MLFlow 서버 연동 테스트는 docs/test_inference_integration.py 사용
"""
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# ── inference 모듈 import 전에 환경변수 설정 (os.environ[] 접근 오류 방지)
os.environ["MLFLOW_TRACKING_URI"]      = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_USERNAME"] = "test"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "test"
os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("S3_BUCKET",  "test-bucket")

from fastapi.testclient import TestClient
from inference import app, model_store, preprocess, FEATURE_COLS


# ── Mock 모델 팩토리
def _clf(proba=(0.22, 0.78), cls=1):
    # native XGBClassifier mock — predict / predict_proba 직접 호출
    m = MagicMock()
    m.predict.return_value = np.array([cls])
    m.predict_proba.return_value = np.array([list(proba)])
    return m

def _reg():
    m = MagicMock()
    m.predict.return_value = np.array([5.43])
    return m

def _le_target(label="Y"):
    # classes_ 는 실제 list (index("Y") 호출 가능해야 함)
    m = MagicMock()
    m.classes_ = np.array(["N", "Y"])
    m.inverse_transform.return_value = [label]
    return m


# ── 샘플 요청
SAMPLE = {
    "gpt_ivt_jg_seq":      "JG_TEST_001",
    "gpt_ivt_mth_cd":      "직접",
    "gpt_ivt_ser_dv_cd":   "오피스",
    "gpt_ivt_tp_cd":       "선순위",
    "gpt_ivt_kd_cd":       "오피스빌딩",
    "gpt_ivt_ara_dv_cd":   "국내",
    "ltv_rte":             65.0,
    "dbt_rpy_coef_rte":    1.35,
    "ln_pd":               24.0,
    "bs_itt":              3.25,
    "gpt_ivt_dlb_rqt_amt": 50000000000.0,
    "gpt_ivt_rmd_lsg_ycn": 3.5,
    "gpt_ivt_etrm_rte":    5.0,
}


@pytest.fixture
def client():
    """
    startup 이벤트의 MLFlow 로드를 no-op 패치 후 Mock 모델을 수동 주입.
    boto3(S3) 호출도 패치하여 AWS 연결 없이 실행.
    """
    with patch.object(model_store, "load"), patch("boto3.client"):
        with TestClient(app) as c:
            model_store.clf       = _clf()
            model_store.reg       = _reg()
            model_store.le_target = _le_target()
            model_store.le_dict   = {}
            model_store.loaded_at = "2026-06-04 10:00:00"
            yield c


# ============================================================
# /health
# ============================================================
class TestHealth:
    def test_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["loaded_at"] == "2026-06-04 10:00:00"

    def test_model_not_loaded(self, client):
        """모델 미로드 상태에서 model_loaded=False 확인"""
        original = model_store.clf
        model_store.clf = None
        assert client.get("/health").json()["model_loaded"] is False
        model_store.clf = original  # 복원


# ============================================================
# /model/info
# ============================================================
class TestModelInfo:
    def test_returns_model_names(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        assert body["cls_model"] == "invest-crel-classification"
        assert body["reg_model"] == "invest-crel-regression"
        assert body["alias"]     == "champion"
        assert "mlflow_uri" in body

    def test_loaded_at(self, client):
        body = client.get("/model/info").json()
        assert body["loaded_at"] == "2026-06-04 10:00:00"


# ============================================================
# /predict
# ============================================================
class TestPredict:
    def test_full_response(self, client):
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 200
        body = r.json()
        assert body["invest_yn"] in ("Y", "N")
        assert 0.0 <= body["invest_prob"] <= 1.0
        assert isinstance(body["fair_rate"], float)
        assert body["gpt_ivt_jg_seq"] == "JG_TEST_001"
        assert "@champion" in body["cls_model"]
        assert "@champion" in body["reg_model"]

    def test_invest_yn_y(self, client):
        """확률 0.78 → invest_yn = Y"""
        r = client.post("/predict", json=SAMPLE)
        body = r.json()
        assert body["invest_prob"] == 0.78
        assert body["invest_yn"] == "Y"

    def test_fair_rate_value(self, client):
        """회귀 모델 출력 5.43 반영 확인"""
        r = client.post("/predict", json=SAMPLE)
        assert r.json()["fair_rate"] == 5.43

    def test_minimal_request(self, client):
        """gpt_ivt_jg_seq 만으로도 200 응답 (나머지 Optional)"""
        r = client.post("/predict", json={"gpt_ivt_jg_seq": "JG_MIN"})
        assert r.status_code == 200

    def test_missing_required_field_422(self, client):
        """gpt_ivt_jg_seq 누락 → Pydantic 422"""
        r = client.post("/predict", json={"ltv_rte": 65.0})
        assert r.status_code == 422

    def test_model_not_loaded_503(self, client):
        """모델 미로드 → 503"""
        original = model_store.clf
        model_store.clf = None
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 503
        assert "미로드" in r.json()["detail"]
        model_store.clf = original

    def test_invest_yn_n_when_prob_low(self, client):
        """클래스 0 예측 → invest_yn = N (le_target 없을 때)"""
        model_store.le_target = None
        low_clf = _clf(proba=(0.70, 0.30), cls=0)
        original = model_store.clf
        model_store.clf = low_clf
        body = client.post("/predict", json=SAMPLE).json()
        assert body["invest_yn"] == "N"
        assert body["invest_prob"] == 0.30
        model_store.clf = original
        model_store.le_target = _le_target()


# ============================================================
# /model/reload
# ============================================================
class TestModelReload:
    def test_reload_success(self, client):
        with patch.object(model_store, "load"):
            r = client.post("/model/reload")
            assert r.status_code == 200
            assert r.json()["status"] == "reloaded"
            assert "loaded_at" in r.json()


# ============================================================
# preprocess() 직접 테스트
# ============================================================
class TestPreprocess:
    def setup_method(self):
        model_store.le_dict = {}

    def test_output_shape(self):
        df = preprocess(SAMPLE)
        assert df.shape == (1, len(FEATURE_COLS))

    def test_columns_in_order(self):
        df = preprocess(SAMPLE)
        assert list(df.columns) == FEATURE_COLS

    def test_missing_columns_filled_zero(self):
        """요청에 없는 피처 컬럼은 0으로 채워짐"""
        df = preprocess({"gpt_ivt_jg_seq": "JG_EMPTY"})
        assert df.shape == (1, len(FEATURE_COLS))
        assert df["ltv_rte"].iloc[0] == 0.0

    def test_string_numeric_coercion(self):
        """문자열 숫자 → float 변환"""
        data = dict(SAMPLE, ltv_rte="65.5")
        df = preprocess(data)
        assert df["ltv_rte"].iloc[0] == 65.5

    def test_invalid_numeric_becomes_zero(self):
        """변환 불가 문자열 → errors='coerce' → 0"""
        data = dict(SAMPLE, ltv_rte="N/A")
        df = preprocess(data)
        assert df["ltv_rte"].iloc[0] == 0.0

    def test_cat_col_with_label_encoder(self):
        """le_dict 존재 시 LabelEncoder 적용"""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["오피스", "리테일", "물류"])
        model_store.le_dict = {"gpt_ivt_ser_dv_cd": le}
        df = preprocess(dict(SAMPLE, gpt_ivt_ser_dv_cd="오피스"))
        assert isinstance(df["gpt_ivt_ser_dv_cd"].iloc[0], (int, np.integer))
        model_store.le_dict = {}

    def test_unknown_cat_value_fallback(self):
        """LabelEncoder에 없는 값 → classes_[0] 으로 폴백"""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["오피스", "리테일"])
        model_store.le_dict = {"gpt_ivt_ser_dv_cd": le}
        df = preprocess(dict(SAMPLE, gpt_ivt_ser_dv_cd="알수없는값"))
        assert isinstance(df["gpt_ivt_ser_dv_cd"].iloc[0], (int, np.integer))
        model_store.le_dict = {}
