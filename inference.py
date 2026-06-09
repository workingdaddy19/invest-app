# ============================================================
# INVEST 부동산담보대출 투자적격 심사 - 추론 API 서버
# FastAPI + MLFlow Registry 기반
# 배포: EKS Pod (독립 애플리케이션)
# ============================================================

import os
import gc
import json
import logging
import traceback
from datetime import datetime

import boto3
import joblib
import numpy as np
import pandas as pd
import pytz
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
from sklearn.preprocessing import LabelEncoder

# ── 환경 설정
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KST = pytz.timezone('Asia/Seoul')

MLFLOW_URI     = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_USER    = os.environ["MLFLOW_TRACKING_USERNAME"]
MLFLOW_PASS    = os.environ["MLFLOW_TRACKING_PASSWORD"]
AWS_REGION     = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET      = os.getenv("S3_BUCKET", "s3-an2-mlops")
MODEL_CLS_NAME = os.getenv("MODEL_CLS_NAME", "invest-crel-classification")
MODEL_REG_NAME = os.getenv("MODEL_REG_NAME", "invest-crel-regression")
MODEL_ALIAS    = os.getenv("MODEL_ALIAS", "champion")

os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASS
mlflow.set_tracking_uri(MLFLOW_URI)

# ── 피처 정의 (invest_crel_model.ipynb 학습과 동일)
NUMERIC_COLS = [
    "gpt_ivt_trc_pi_rk", "gpt_ivt_dlb_rqt_amt", "ln_pd",
    "ltv_rte", "gpt_ivt_cpt_ern_rte", "gpt_ivt_cpt_ern_pd",
    "gpt_ivt_all_pcm_amt", "gpt_ivt_bdg_scl_txt", "gpt_ivt_nwk_ot_scl_txt",
    "gpt_ivt_cmpi_yr", "dbt_rpy_coef_rte", "gpt_ivt_rmd_lsg_ycn",
    "gpt_ivt_etrm_rte", "gpt_ivt_mkt_avg_etrm_rt",
    "gpt_ivt_ppo_re_amt", "gpt_ivt_mkt_ppo_re_amt",
    "gpt_ivt_mkt_avg_cpt_rte", "gpt_ivt_mkt_avg_dln_amt",
    "gpt_ivt_ln_pfat_txt", "gpt_ivt_te_ppo_amt",
    "gpt_ivt_cpt_reim", "gpt_ivt_appr_evl_ppo_amt",
    "gpt_ivt_rpy_rte", "bs_itt",
]
CAT_COLS = [
    "gpt_ivt_mth_cd", "gpt_ivt_ser_dv_cd", "gpt_ivt_tp_cd",
    "gpt_ivt_str_dv_cd", "gpt_ivt_kd_cd", "gpt_ivt_ara_dv_cd",
    "gpt_ivt_crd_rinf_txt", "gpt_ivt_ecfr_gd_txt",
]
FEATURE_COLS = NUMERIC_COLS + CAT_COLS


# ============================================================
# 모델 스토어 (앱 시작 시 1회 로드 → 메모리 캐싱)
# ============================================================
class ModelStore:
    def __init__(self):
        self.clf = self.reg = self.le_target = None
        self.le_dict = {}
        self.loaded_at = None

    def _uri(self, name):
        return f"models:/{name}@{MODEL_ALIAS}"

    def load(self):
        logger.info("MLFlow 모델 로딩 시작...")
        # native XGBoost 로드 → predict_proba 사용 가능 (XGBClassifier/XGBRegressor)
        self.clf = mlflow.xgboost.load_model(self._uri(MODEL_CLS_NAME))
        self.reg = mlflow.xgboost.load_model(self._uri(MODEL_REG_NAME))
        self._load_encoders()
        self.loaded_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"모델 로드 완료 ({self.loaded_at})")

    def _load_encoders(self):
        """MLFlow Artifact에 encoders/le_dict.pkl, le_target.pkl이 있으면 로드"""
        try:
            client = mlflow.tracking.MlflowClient()
            ver = client.get_model_version_by_alias(MODEL_CLS_NAME, MODEL_ALIAS)
            path = mlflow.artifacts.download_artifacts(run_id=ver.run_id, artifact_path="encoders")
            self.le_dict   = joblib.load(f"{path}/le_dict.pkl")
            self.le_target = joblib.load(f"{path}/le_target.pkl")
            logger.info("LabelEncoder 로드 완료")
        except Exception:
            logger.warning("LabelEncoder artifact 없음 → 해시 기반 정수 변환 사용")
            self.le_dict = {}
            self.le_target = None

    def reload(self):
        logger.info("핫 리로드 시작...")
        self.load()


model_store = ModelStore()


# ============================================================
# FastAPI 앱
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store.load()
    yield

app = FastAPI(
    title="INVEST 투자적격 심사 추론 API",
    description="부동산담보대출 투자적격판단 + 적정금리평가 모델 서빙",
    version="1.0.0",
    lifespan=lifespan,
)


# ── 요청/응답 스키마
class InferenceRequest(BaseModel):
    gpt_ivt_jg_seq:           str
    gpt_fl_nm:                Optional[str]   = None
    gpt_ivt_mth_cd:           Optional[str]   = None
    gpt_ivt_ser_dv_cd:        Optional[str]   = None
    gpt_ivt_tp_cd:            Optional[str]   = None
    gpt_ivt_str_dv_cd:        Optional[str]   = None
    gpt_ivt_kd_cd:            Optional[str]   = None
    gpt_ivt_ara_dv_cd:        Optional[str]   = None
    gpt_ivt_crd_rinf_txt:     Optional[str]   = None
    gpt_ivt_ecfr_gd_txt:      Optional[str]   = None
    gpt_ivt_trc_pi_rk:        Optional[float] = None
    gpt_ivt_dlb_rqt_amt:      Optional[float] = None
    ln_pd:                    Optional[float] = None
    ltv_rte:                  Optional[float] = None
    gpt_ivt_cpt_ern_rte:      Optional[float] = None
    gpt_ivt_cpt_ern_pd:       Optional[float] = None
    gpt_ivt_all_pcm_amt:      Optional[float] = None
    gpt_ivt_bdg_scl_txt:      Optional[float] = None
    gpt_ivt_nwk_ot_scl_txt:   Optional[float] = None
    gpt_ivt_cmpi_yr:          Optional[float] = None
    dbt_rpy_coef_rte:         Optional[float] = None
    gpt_ivt_rmd_lsg_ycn:      Optional[float] = None
    gpt_ivt_etrm_rte:         Optional[float] = None
    gpt_ivt_mkt_avg_etrm_rt:  Optional[float] = None
    gpt_ivt_ppo_re_amt:       Optional[float] = None
    gpt_ivt_mkt_ppo_re_amt:   Optional[float] = None
    gpt_ivt_mkt_avg_cpt_rte:  Optional[float] = None
    gpt_ivt_mkt_avg_dln_amt:  Optional[float] = None
    gpt_ivt_ln_pfat_txt:      Optional[float] = None
    gpt_ivt_te_ppo_amt:       Optional[float] = None
    gpt_ivt_cpt_reim:         Optional[float] = None
    gpt_ivt_appr_evl_ppo_amt: Optional[float] = None
    gpt_ivt_rpy_rte:          Optional[float] = None
    bs_itt:                   Optional[float] = None


class InferenceResponse(BaseModel):
    request_id:      str
    gpt_ivt_jg_seq:  str
    invest_yn:       str    # Y / N
    invest_prob:     float  # 승인 확률 (0~1)
    fair_rate:       float  # 적정 금리 (%)
    cls_model:       str
    reg_model:       str
    inferred_at:     str


# ── 전처리
def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # 누락 컬럼 0으로 초기화 (요청에 없는 피처 안전 처리)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)  # 실운영: 학습셋 중앙값 주입 권장

    for col in CAT_COLS:
        raw = df[col].iloc[0]
        val = str(raw) if raw is not None else "UNKNOWN"
        if col in model_store.le_dict:
            le = model_store.le_dict[col]
            val = val if val in le.classes_ else le.classes_[0]
            df[col] = le.transform([val])
        else:
            df[col] = hash(val) % 100

    return df[FEATURE_COLS]


def _s3_log(request_id: str, stage: str, status: str, data: dict):
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        yyyymmdd = datetime.now(KST).strftime("%Y%m%d")
        key = f"invest-model-result/LOG/{yyyymmdd}/{request_id}_{stage}_{status}.json".upper()
        body = json.dumps({"request_id": request_id, "stage": stage, "status": status,
                           "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"), **data},
                          ensure_ascii=False, indent=2)
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body)
    except Exception as e:
        logger.warning(f"S3 로그 실패: {e}")


# ── 엔드포인트


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_store.clf is not None, "loaded_at": model_store.loaded_at}


@app.get("/model/info")
def model_info():
    return {"mlflow_uri": MLFLOW_URI, "cls_model": MODEL_CLS_NAME,
            "reg_model": MODEL_REG_NAME, "alias": MODEL_ALIAS, "loaded_at": model_store.loaded_at}


@app.post("/model/reload")
def model_reload():
    """champion alias 변경 후 재배포 없이 새 버전 로드"""
    try:
        model_store.reload()
        return {"status": "reloaded", "loaded_at": model_store.loaded_at}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    request_id = datetime.now(KST).strftime("%Y%m%d%H%M%S%f")
    logger.info(f"[{request_id}] 추론 요청: {req.gpt_ivt_jg_seq}")
    _s3_log(request_id, "inference", "start", {"gpt_ivt_jg_seq": req.gpt_ivt_jg_seq})

    try:
        if model_store.clf is None:
            raise HTTPException(status_code=503, detail="모델 미로드 상태")

        X = preprocess(req.model_dump())

        # 분류: 클래스 예측 + 적격(Y) 확률
        cls_pred = model_store.clf.predict(X)
        proba = model_store.clf.predict_proba(X)[0]

        if model_store.le_target is not None:
            classes = list(model_store.le_target.classes_)
            invest_yn = model_store.le_target.inverse_transform(cls_pred.astype(int))[0]
            y_idx = classes.index("Y") if "Y" in classes else (len(proba) - 1)
        else:
            invest_yn = "Y" if int(cls_pred[0]) == 1 else "N"
            y_idx = 1 if len(proba) > 1 else 0
        invest_prob = float(proba[y_idx])

        # 회귀: 적정 금리
        fair_rate = float(model_store.reg.predict(X)[0])

        result = {
            "request_id":     request_id,
            "gpt_ivt_jg_seq": req.gpt_ivt_jg_seq,
            "invest_yn":      invest_yn,
            "invest_prob":    round(invest_prob, 4),
            "fair_rate":      round(fair_rate, 4),
            "cls_model":      f"{MODEL_CLS_NAME}@{MODEL_ALIAS}",
            "reg_model":      f"{MODEL_REG_NAME}@{MODEL_ALIAS}",
            "inferred_at":    datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        }
        _s3_log(request_id, "inference", "success", result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        _s3_log(request_id, "inference", "error",
                {"error_message": str(e), "stack_trace": traceback.format_exc()})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def test_ui():
    """내장 테스트 UI — Pod 브라우저 접근 시 바로 추론 테스트 가능"""
    return HTMLResponse(content=_UI_HTML)


_UI_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>INVEST 추론 테스트</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f1f5f9;color:#1e293b;padding:24px}
h1{font-size:18px;font-weight:700;margin-bottom:4px}
.sub{font-size:12px;color:#64748b;margin-bottom:20px}
.card{background:#fff;border-radius:10px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card-title{font-size:13px;font-weight:700;color:#475569;margin-bottom:14px;text-transform:uppercase;letter-spacing:.05em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
label{display:block;font-size:11px;color:#64748b;margin-bottom:3px}
input{width:100%;padding:7px 10px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;outline:none}
input:focus{border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,.15)}
.req{color:#ef4444}
.row{display:flex;gap:10px;align-items:center;margin-top:14px}
.btn{padding:9px 22px;border:none;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer}
.btn-primary{background:#6366f1;color:#fff}
.btn-primary:hover{background:#4f46e5}
.btn-secondary{background:#e2e8f0;color:#475569}
.btn-secondary:hover{background:#cbd5e1}
#status{font-size:12px;color:#64748b}
/* result */
.result-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}
.result-box{text-align:center;padding:16px;border-radius:8px;background:#f8fafc}
.result-label{font-size:11px;color:#64748b;margin-bottom:6px}
.result-value{font-size:26px;font-weight:800}
.yn-y{color:#10b981} .yn-n{color:#ef4444}
.val-prob{color:#6366f1} .val-rate{color:#0891b2}
.meta{font-size:11px;color:#94a3b8;border-top:1px solid #f1f5f9;padding-top:10px}
.err{border-left:3px solid #ef4444;padding:10px 14px}
.err-title{font-size:12px;font-weight:700;color:#ef4444;margin-bottom:4px}
.err-body{font-size:11px;color:#64748b;font-family:monospace;white-space:pre-wrap}
/* health badge */
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;padding:4px 10px;border-radius:20px;font-weight:600}
.badge-ok{background:#d1fae5;color:#065f46}
.badge-warn{background:#fef3c7;color:#92400e}
.badge-err{background:#fee2e2;color:#991b1b}
</style>
</head>
<body>
<h1>INVEST 투자적격 심사 — 추론 테스트</h1>
<p class="sub">내장 테스트 UI · invest-inference Pod</p>

<!-- 상태 카드 -->
<div class="card" id="health-card" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
  <div>
    <div style="font-size:11px;color:#94a3b8;margin-bottom:2px">MLflow URI</div>
    <div style="font-size:12px;font-family:monospace;color:#475569" id="mlflow-uri">확인 중...</div>
  </div>
  <span id="health-badge" class="badge badge-warn">● 확인 중</span>
</div>

<!-- 입력 폼 -->
<div class="card">
  <div class="card-title">입력 항목</div>
  <div class="grid">
    <div><label>심사일련번호 <span class="req">*</span></label><input id="jg_seq" value="JG_TEST_001" placeholder="JG20260604001"></div>
    <div><label>LTV (%)</label><input id="ltv_rte" type="number" step="0.1" value="65.0"></div>
    <div><label>DSCR</label><input id="dbt_rpy_coef_rte" type="number" step="0.01" value="1.35"></div>
    <div><label>대출기간 (월)</label><input id="ln_pd" type="number" value="24"></div>
    <div><label>기준금리 (%)</label><input id="bs_itt" type="number" step="0.01" value="3.25"></div>
    <div><label>공실률 (%)</label><input id="gpt_ivt_etrm_rte" type="number" step="0.1" value="5.0"></div>
    <div><label>잔여임대차기간 (년)</label><input id="gpt_ivt_rmd_lsg_ycn" type="number" step="0.1" value="3.5"></div>
    <div><label>심의요청금액 (원)</label><input id="gpt_ivt_dlb_rqt_amt" type="number" value="50000000000"></div>
    <div><label>Debt Yield (%)</label><input id="gpt_ivt_ln_pfat_txt" type="number" step="0.1" value="8.5"></div>
    <div><label>Cap Rate 시장평균 (%)</label><input id="gpt_ivt_mkt_avg_cpt_rte" type="number" step="0.1" value="4.3"></div>
    <div><label>물건구분</label><input id="gpt_ivt_kd_cd" value="오피스빌딩"></div>
    <div><label>투자섹터</label><input id="gpt_ivt_ser_dv_cd" value="오피스"></div>
    <div><label>투자유형</label><input id="gpt_ivt_tp_cd" value="선순위"></div>
    <div><label>투자방법</label><input id="gpt_ivt_mth_cd" value="직접"></div>
    <div><label>지역구분</label><input id="gpt_ivt_ara_dv_cd" value="국내"></div>
  </div>
  <div class="row">
    <button class="btn btn-primary" onclick="runPredict()">추론 실행</button>
    <button class="btn btn-secondary" onclick="reset()">초기화</button>
    <span id="status"></span>
  </div>
</div>

<!-- 결과 -->
<div class="card" id="result" style="display:none">
  <div class="card-title">추론 결과</div>
  <div class="result-grid">
    <div class="result-box"><div class="result-label">투자적격 여부</div><div class="result-value" id="r-yn"></div></div>
    <div class="result-box"><div class="result-label">적격 확률</div><div class="result-value val-prob" id="r-prob"></div></div>
    <div class="result-box"><div class="result-label">AI 추천 금리</div><div class="result-value val-rate" id="r-rate"></div></div>
  </div>
  <div class="meta" id="r-meta"></div>
</div>

<!-- 오류 -->
<div class="card err" id="error" style="display:none">
  <div class="err-title">오류</div>
  <div class="err-body" id="err-msg"></div>
</div>

<script>
window.onload = async () => {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    const badge = document.getElementById('health-badge');
    if (d.model_loaded) {
      badge.className = 'badge badge-ok';
      badge.textContent = '● 모델 로드 완료  ' + (d.loaded_at || '');
    } else {
      badge.className = 'badge badge-warn';
      badge.textContent = '● 서버 응답 / 모델 미로드';
    }
  } catch {
    const badge = document.getElementById('health-badge');
    badge.className = 'badge badge-err';
    badge.textContent = '● 서버 응답 없음';
  }
  try {
    const r = await fetch('/model/info');
    const d = await r.json();
    document.getElementById('mlflow-uri').textContent = d.mlflow_uri;
  } catch {}
};

async function runPredict() {
  const jgSeq = document.getElementById('jg_seq').value.trim();
  if (!jgSeq) { alert('심사일련번호는 필수입니다.'); return; }

  const f = (id) => { const v = document.getElementById(id).value; return v !== '' ? parseFloat(v) : null; };
  const s = (id) => document.getElementById(id).value.trim() || null;

  const payload = {
    gpt_ivt_jg_seq:         jgSeq,
    ltv_rte:                f('ltv_rte'),
    dbt_rpy_coef_rte:       f('dbt_rpy_coef_rte'),
    ln_pd:                  f('ln_pd'),
    bs_itt:                 f('bs_itt'),
    gpt_ivt_etrm_rte:       f('gpt_ivt_etrm_rte'),
    gpt_ivt_rmd_lsg_ycn:    f('gpt_ivt_rmd_lsg_ycn'),
    gpt_ivt_dlb_rqt_amt:    f('gpt_ivt_dlb_rqt_amt'),
    gpt_ivt_ln_pfat_txt:    f('gpt_ivt_ln_pfat_txt'),
    gpt_ivt_mkt_avg_cpt_rte:f('gpt_ivt_mkt_avg_cpt_rte'),
    gpt_ivt_kd_cd:          s('gpt_ivt_kd_cd'),
    gpt_ivt_ser_dv_cd:      s('gpt_ivt_ser_dv_cd'),
    gpt_ivt_tp_cd:          s('gpt_ivt_tp_cd'),
    gpt_ivt_mth_cd:         s('gpt_ivt_mth_cd'),
    gpt_ivt_ara_dv_cd:      s('gpt_ivt_ara_dv_cd'),
  };

  hide('result'); hide('error');
  set('status', '추론 중...');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await res.json();
    if (!res.ok) { showErr(d.detail || JSON.stringify(d)); return; }

    const yn = d.invest_yn;
    const ynEl = document.getElementById('r-yn');
    ynEl.textContent = yn === 'Y' ? '✅ 적격' : '❌ 부적격';
    ynEl.className = 'result-value ' + (yn === 'Y' ? 'yn-y' : 'yn-n');
    document.getElementById('r-prob').textContent = (d.invest_prob * 100).toFixed(1) + '%';
    document.getElementById('r-rate').textContent = d.fair_rate.toFixed(4) + '%';
    document.getElementById('r-meta').innerHTML =
      'ID: <b>' + d.request_id + '</b> &nbsp;|&nbsp; ' +
      d.cls_model + ' &nbsp;|&nbsp; ' + d.reg_model +
      ' &nbsp;|&nbsp; ' + d.inferred_at;
    show('result');
  } catch(e) {
    showErr(e.message);
  } finally {
    set('status', '');
  }
}

function showErr(msg) { document.getElementById('err-msg').textContent = msg; show('error'); }
function show(id) { document.getElementById(id).style.display = 'block'; }
function hide(id) { document.getElementById(id).style.display = 'none'; }
function set(id, v) { document.getElementById(id).textContent = v; }
function reset() {
  ['ltv_rte','dbt_rpy_coef_rte','ln_pd','bs_itt','gpt_ivt_etrm_rte',
   'gpt_ivt_rmd_lsg_ycn','gpt_ivt_dlb_rqt_amt','gpt_ivt_ln_pfat_txt',
   'gpt_ivt_mkt_avg_cpt_rte','jg_seq'].forEach(id => document.getElementById(id).value = '');
  hide('result'); hide('error');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8080, reload=False)
