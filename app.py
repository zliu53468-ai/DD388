from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# 模型檔案路徑
XGB_MODEL_PATH = "baccarat_ml_model_xgboost.json"
LGB_MODEL_PATH = "baccarat_ml_model_lightgbm.txt"
LABEL_ENCODER_PATH = "baccarat_ml_label_encoder.joblib"
SCALER_PATH = "baccarat_ml_feature_scaler.joblib"

# 後端歷史紀錄
game_history_backend = []

# 嘗試載入模型
def load_models():
    global xgb_model, lgb_model, label_encoder, scaler
    try:
        # 使用 os.path.join 來確保路徑正確
        xgb_model = xgb.Booster()
        xgb_model.load_model(os.path.join(os.path.dirname(__file__), XGB_MODEL_PATH))
        lgb_model = lgb.Booster(model_file=os.path.join(os.path.dirname(__file__), LGB_MODEL_PATH))
        label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), LABEL_ENCODER_PATH))
        scaler = joblib.load(os.path.join(os.path.dirname(__file__), SCALER_PATH))
        print("✅ 模型載入成功")
    except Exception as e:
        print("⚠️ 模型尚未訓練或載入失敗：", e)
        xgb_model = None
        lgb_model = None
        label_encoder = None
        scaler = None

load_models()

# === 大路生成 ===
def generate_big_road(history):
    rows, cols = 6, 100
    road_map = [[None for _ in range(cols)] for _ in range(rows)]
    col = row = 0
    last_bp, last_bp_pos = None, None

    for result in history:
        if result == 'T':
            if last_bp_pos:
                r, c = last_bp_pos
                if road_map[r][c] is None:
                    road_map[r][c] = {"result": last_bp, "ties": 1}
                else:
                    road_map[r][c]["ties"] += 1
            continue

        if last_bp is None:
            road_map[row][col] = {"result": result, "ties": 0}
            last_bp, last_bp_pos = result, (row, col)
            continue

        if result == last_bp:
            if row + 1 < rows and road_map[row + 1][col] is None:
                row += 1
            else:
                col += 1
        else:
            if col + 1 < cols:
                row = 0
                col += 1

        road_map[row][col] = {"result": result, "ties": 0}
        last_bp, last_bp_pos = result, (row, col)

    max_col = max(c for c in range(cols) if any(road_map[r][c] for r in range(rows))) if any(any(road_map[r][c] for r in range(rows)) for c in range(cols)) else -1
    if max_col == -1:
        return []

    trimmed_map = [row[:max_col + 1] for row in road_map]

    output = []
    for r in range(rows):
        for c, cell in enumerate(trimmed_map[r]):
            if cell:
                output.append({
                    "row": r, "col": c,
                    "result": cell["result"],
                    "ties": cell["ties"]
                })
    return output

# === API ===
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "xgb_loaded": xgb_model is not None,
        "lgb_loaded": lgb_model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not all([xgb_model, lgb_model, label_encoder, scaler]):
        return jsonify({"error": "模型尚未訓練或載入失敗"}), 400

    req_data = request.get_json()
    if not req_data or "features" not in req_data:
        return jsonify({"error": "缺少 features"}), 400

    features = np.array([req_data["features"]], dtype=np.float64) # 修正：確保型別正確
    features_scaled = scaler.transform(features)

    try:
        xgb_pred = xgb_model.predict(xgb.DMatrix(features_scaled))
        lgb_pred = lgb_model.predict(features_scaled)
    except Exception as e:
        return jsonify({"error": f"預測錯誤: {str(e)}"}), 500

    xgb_label = label_encoder.inverse_transform([np.argmax(xgb_pred)])[0]
    lgb_label = label_encoder.inverse_transform([np.argmax(lgb_pred)])[0]

    return jsonify({
        "xgboost": xgb_label,
        "lightgbm": lgb_label
    })

@app.route("/train", methods=["POST"])
def train():
    global xgb_model, lgb_model, label_encoder, scaler
    req_data = request.get_json()
    if not req_data or "data" not in req_data:
        return jsonify({"error": "缺少訓練資料"}), 400

    X = np.array([d["features"] for d in req_data["data"]], dtype=np.float64) # 修正：確保型別正確
    y = np.array([d["label"] for d in req_data["data"]])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb_model = xgb.train({}, xgb.DMatrix(X_scaled, label=y_encoded), num_boost_round=10)
    lgb_model = lgb.train({}, lgb.Dataset(X_scaled, label=y_encoded), num_boost_round=10)

    xgb_model.save_model(XGB_MODEL_PATH)
    lgb_model.save_model(LGB_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    load_models()
    return jsonify({"message": "模型訓練完成"})

@app.route("/api/history", methods=["GET", "POST", "DELETE"])
def handle_history_api():
    global game_history_backend
    if request.method == "GET":
        return jsonify({
            "history": game_history_backend,
            "big_road": generate_big_road(game_history_backend)
        })
    elif request.method == "POST":
        req_data = request.get_json()
        if not req_data or "history" not in req_data:
            return jsonify({"error": "缺少 history"}), 400
        game_history_backend = [h.upper() for h in req_data["history"]]
        return jsonify({"message": "歷史已更新"})
    elif request.method == "DELETE":
        game_history_backend.clear()
        return jsonify({"message": "歷史已清空"})

# 讓 Gunicorn 或其他 WSGI 伺服器來處理啟動
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)
