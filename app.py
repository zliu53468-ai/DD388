from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from collections import defaultdict
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

app = Flask(__name__)

# ====== CORS 配置 (已更新為支援 GitHub Pages 的明確來源) ======
# 請將 'https://your-github-username.github.io' 替換為您實際的 GitHub Pages 網址！
CORS(app, origins=[
    "https://cdpn.io",
    "https://*.codepen.io",
    "https://*.codesandbox.io",
    "https://zliu53468-ai.github.io", # <--- 已替換為您的 GitHub Pages 網址
    "http://localhost:*",
    "http://127.0.0.1:*"
], methods=["GET", "POST", "DELETE", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])


# 全域變數 - 模型和標籤編碼器
ml_model_xgboost = None
ml_model_lightgbm = None
ml_label_encoder = None
feature_scaler = None

# 模型檔案路徑
MODEL_FILE_XGBOOST = 'baccarat_ml_model_xgboost.json'
MODEL_FILE_LIGHTGBM = 'baccarat_ml_model_lightgbm.txt'
MODEL_FILE_ENCODER = 'baccarat_ml_label_encoder.joblib'
MODEL_FILE_SCALER = 'baccarat_ml_feature_scaler.joblib'

LOOK_BACK = 30 # 增加回溯步長以捕捉更多模式，特別是路紙模式
game_history_backend = [] # 用於模擬後端歷史數據的全局變量
ai_prediction_outcomes_backend = [] # 新增: 儲存AI預測在該局是否正確 (True/False)


# --- 馬可夫鏈模型 ---
MARKOV_LOOK_BACK = 5 # 調整馬可夫鏈的回溯深度
markov_chain_model = defaultdict(lambda: defaultdict(int))

def update_markov_chain(history_data):
    """根據歷史數據更新馬可夫鏈模型。"""
    global markov_chain_model
    markov_chain_model = defaultdict(lambda: defaultdict(int)) # 每次訓練前重置

    if len(history_data) < MARKOV_LOOK_BACK + 1:
        return

    for i in range(len(history_data) - MARKOV_LOOK_BACK):
        current_state = tuple(history_data[i : i + MARKOV_LOOK_BACK])
        next_state = history_data[i + MARKOV_LOOK_BACK]
        markov_chain_model[current_state][next_state] += 1

def predict_markov_chain(history_slice):
    """使用馬可夫鏈模型預測下一個結果的機率。"""
    if len(history_slice) < MARKOV_LOOK_BACK:
        return {'B': 1/3, 'P': 1/3, 'T': 1/3} # 數據不足時返回均等機率

    current_state = tuple(history_slice[-MARKOV_LOOK_BACK:])
    
    if current_state not in markov_chain_model:
        return {'B': 1/3, 'P': 1/3, 'T': 1/3} # 未知狀態返回均等機率

    transitions = markov_chain_model[current_state]
    total_transitions = sum(transitions.values())

    if total_transitions == 0:
        return {'B': 1/3, 'P': 1/3, 'T': 1/3}

    probabilities = {
        outcome: count / total_transitions
        for outcome, count in transitions.items()
    }

    # 確保所有結果都有機率，如果沒有則補 0
    for outcome in ['B', 'P', 'T']:
        if outcome not in probabilities:
            probabilities[outcome] = 0.0

    return probabilities

# --- 百家樂俚語特徵工程函數 ---
def _get_column(road_map, row, col):
    """從路單地圖中獲取指定位置的結果，如果超出範圍則返回None。"""
    if 0 <= row < len(road_map) and 0 <= col < len(road_map[0]):
        return road_map[row][col]
    return None

def _get_big_road_map(history_data):
    """
    根據歷史數據生成大路地圖，並正確處理和局。
    每個單元格現在是一個字典，包含 'result' (B/P) 和 'ties' (和局數)。
    """
    try:
        # 初始化一個 6 行 N 列的二維列表，每個元素是一個字典 {'result': '', 'ties': 0}
        road_map = [[{'result': '', 'ties': 0} for _ in range(100)] for _ in range(6)]
        col = 0
        row = 0
        last_bp_result = '' # 記錄上一個非和局的結果
        last_bp_position = None # 新增：追蹤最後一個B/P的位置 (row, col)

        if not history_data:
            return road_map

        for i in range(len(history_data)):
            current_result = history_data[i]

            if current_result == 'T':
                # 和局處理：在最後一個B/P上增加和局計數
                if last_bp_position:
                    r, c = last_bp_position
                    road_map[r][c]['ties'] += 1
                continue

            # 處理 B 或 P - 重置邏輯修正
            if not last_bp_result: # 第一個 B 或 P
                # 重置到起始位置
                col = 0
                row = 0
            elif current_result == last_bp_result: # 連續相同
                if row < 5: # 向下移動
                    row += 1
                else: # 換到下一列, 繼續在第6行(索引5)繪製，這是標準龍尾
                    col += 1
                    row = 5 # 保持在底部
            else: # 結果改變
                col += 1
                row = 0
            
            # 確保不超出範圍
            if col >= 100:
                break
            
            road_map[row][col]['result'] = current_result
            last_bp_result = current_result
            last_bp_position = (row, col)
        
        # 清理多餘的空列
        final_cols = 0
        for c_idx in range(100):
            col_is_empty = True
            for r_idx in range(6):
                if road_map[r_idx][c_idx]['result'] != '':
                    col_is_empty = False
                    break
            if not col_is_empty:
                final_cols = c_idx + 1
        
        return [row[:final_cols] for row in road_map]
    except Exception as e:
        print(f"生成大路地圖錯誤: {e}")
        # 返回空的大路或處理錯誤
        return [[{'result': '', 'ties': 0}] for _ in range(6)]


def is_pattern_long_run(history_slice, min_len=5):
    """判斷是否為「長龍」：連續出現相同結果（B或P）。"""
    if len(history_slice) < min_len:
        return 0
    
    # 過濾和局，只考慮 B/P 序列
    filtered_history = [r for r in history_slice if r != 'T']
    if len(filtered_history) < min_len:
        return 0

    last_outcome = filtered_history[-1]
    
    count = 0
    for i in range(1, len(filtered_history) + 1):
        if filtered_history[-i] == last_outcome:
            count += 1
        else:
            break
    return 1 if count >= min_len else 0


def is_pattern_single_jump(history_slice, min_len=4):
    """判斷是否為「單跳」：B P B P 或 P B P B。"""
    if len(history_slice) < min_len:
        return 0

    filtered_history = [r for r in history_slice if r != 'T']
    if len(filtered_history) < min_len:
        return 0

    is_single_jump = True
    # 檢查是否為 A B A B ...
    for i in range(1, min_len):
        if filtered_history[-i] == filtered_history[-(i+1)]:
            is_single_jump = False
            break
    
    return 1 if is_single_jump else 0


def is_pattern_double_jump(history_slice, min_len=6):
    """判斷是否為「雙跳」：B B P P B B 或 P P B B P P。"""
    if len(history_slice) < min_len:
        return 0

    filtered_history = [r for r in history_slice if r != 'T']
    if len(filtered_history) < min_len:
        return 0

    pattern = filtered_history[-min_len:]
    if len(pattern) != min_len: # should already be handled by filtered_history check
        return 0

    # 模式: X X Y Y X X ...
    if pattern[0] == pattern[1] and \
       pattern[2] == pattern[3] and \
       pattern[4] == pattern[5] and \
       pattern[0] == pattern[4] and \
       pattern[1] == pattern[5] and \
       pattern[0] != pattern[2]:
        return 1
    return 0


def is_pattern_two_in_a_row_one_off(history_slice, min_len=6):
    """判斷是否為「一房兩廳」：B B P B B P 或 P P B P P B。"""
    if len(history_slice) < min_len:
        return 0

    filtered_history = [r for r in history_slice if r != 'T']
    if len(filtered_history) < min_len:
        return 0

    pattern = filtered_history[-min_len:]
    if len(pattern) != min_len: # should already be handled by filtered_history check
        return 0

    # 檢查是否為 A A B A A B
    if pattern[0] == pattern[1] and \
       pattern[0] == pattern[3] and \
       pattern[0] == pattern[4] and \
       pattern[2] == pattern[5] and \
       pattern[0] != pattern[2]:
        return 1
    return 0


def get_virtual_road_features(history_data):
    """根據大路生成大眼仔、小路、曱甴路的虛擬特徵。"""
    features = {
        'big_road_cols': 0,  # 大路總列數
        'last_big_road_length': 0,  # 大路最後一列的長度
        'big_road_alternating_pattern': 0,  # 大路是否有交替模式 (例如 B P B P)
        'big_road_double_pattern': 0,  # 大路是否有雙連模式 (例如 BB PP BB)
        'big_road_flat_line_pattern': 0,  # 大路是否有齊腳路 (三列或更多都一樣長)
        'big_road_straight_line_pattern': 0,  # 大路是否有長條路 (超過三條長龍)
    }

    if not history_data:
        return features

    # 過濾掉 'T' 來構建大路
    filtered_history = [r for r in history_data if r != 'T']
    if not filtered_history:
        return features

    big_road_cols_data = []  # 儲存每一列的結果
    current_col = []
    for i, result in enumerate(filtered_history):
        if not current_col:
            current_col.append(result)
        elif result == current_col[-1]:
            current_col.append(result)
        else:
            big_road_cols_data.append(current_col)
            current_col = [result]
    if current_col:
        big_road_cols_data.append(current_col)

    features['big_road_cols'] = len(big_road_cols_data)
    if big_road_cols_data:
        features['last_big_road_length'] = len(big_road_cols_data[-1])

    if len(big_road_cols_data) >= 2:
        # 檢查交替模式 (B P B P...)
        alternating = True
        for i in range(len(big_road_cols_data) - 1):
            if big_road_cols_data[i][0] == big_road_cols_data[i+1][0]:
                alternating = False
                break
        features['big_road_alternating_pattern'] = 1 if alternating else 0

    if len(big_road_cols_data) >= 4:
        # 檢查雙連模式 (BB PP BB...)
        # 這裡的邏輯需要確保不會因為索引超出範圍而崩潰
        # 並且檢查模式的正確性，例如 AABBAA
        if (len(big_road_cols_data) >= 3 and
            len(big_road_cols_data[-1]) == 2 and
            len(big_road_cols_data[-2]) == 2 and
            len(big_road_cols_data[-3]) == 2 and
            big_road_cols_data[-1][0] == big_road_cols_data[-3][0] and
                big_road_cols_data[-1][0] != big_road_cols_data[-2][0]):
            features['big_road_double_pattern'] = 1

    # 檢查齊腳路 (至少三列，且長度相似)
    if len(big_road_cols_data) >= 3:
        all_lengths = [len(col) for col in big_road_cols_data[-3:]]
        if max(all_lengths) - min(all_lengths) <= 1:  # 允許1的差異
            features['big_road_flat_line_pattern'] = 1

    # 檢查長條路 (例如連續超過3條長龍)
    if len(big_road_cols_data) >= 4:
        straight_line = True
        for i in range(len(big_road_cols_data) - 3):
            # 判斷是否為長龍
            if len(big_road_cols_data[i]) < 5:  # 定義長龍至少5個
                straight_line = False
                break
            # 判斷是否結果相同
            if big_road_cols_data[i][0] != big_road_cols_data[i+1][0] or \
               big_road_cols_data[i+1][0] != big_road_cols_data[i+2][0] or \
               big_road_cols_data[i+2][0] != big_road_cols_data[i+3][0]:
                straight_line = False
                break
        features['big_road_straight_line_pattern'] = 1 if straight_line else 0

    return features


# --- 特徵工程函數 (包含百家樂俚語特徵) ---
def encode_result_for_ml(result):
    """將 B, P, T 編碼為 One-Hot 格式。"""
    if result == 'B':
        return [1, 0, 0]
    if result == 'P':
        return [0, 1, 0]
    if result == 'T':
        return [0, 0, 1]
    return [0, 0, 0]


def numerical_encode_result(result):
    """將 B, P, T 編碼為數值用於統計計算。"""
    if result == 'B':
        return 1
    if result == 'P':
        return -1
    if result == 'T':
        return 0
    return 0


def get_outcome_ratios(history_slice, outcome_type):
    """計算特定結果的比例。"""
    if not history_slice:
        return 0.0
    count = history_slice.count(outcome_type)
    return count / len(history_slice)


def get_recent_trend(history_slice, window=10):
    """計算最近窗口的趨勢。"""
    if len(history_slice) < window:
        return 0.0

    recent = history_slice[-window:]
    b_count = recent.count('B')
    p_count = recent.count('P')

    if b_count + p_count == 0:
        return 0.0
    return (b_count - p_count) / (b_count + p_count)


def get_streak_length(history_slice):
    """計算當前連勝長度。"""
    if not history_slice:
        return 0

    current = history_slice[-1]
    if current not in ['B', 'P']:
        return 0

    streak = 1
    for i in range(2, len(history_slice) + 1):
        if history_slice[-i] == current:
            streak += 1
        else:
            break
    return streak


def get_gap_since_last(history_slice, outcome):
    """計算距離上一次出現特定結果的局數。"""
    if outcome not in history_slice:
        return len(history_slice)

    for i in range(len(history_slice) - 1, -1, -1):
        if history_slice[i] == outcome:
            return len(history_slice) - i - 1
    return len(history_slice)  # 不應該到達這裡


def prepare_simplified_features(history_data, look_back=LOOK_BACK):
    """準備簡化的特徵集，包含百家樂俚語特徵和路單分析特徵。"""
    if len(history_data) < look_back:
        return None

    features = []

    # 1. 最近序列的 one-hot 編碼（扁平化）
    recent_sequence = history_data[-look_back:]
    encoded_sequence_flat = []
    for res in recent_sequence:
        encoded_sequence_flat.extend(encode_result_for_ml(res))
    features.extend(encoded_sequence_flat)

    # 2. 整體比例特徵
    features.append(get_outcome_ratios(history_data, 'B'))
    features.append(get_outcome_ratios(history_data, 'P'))
    features.append(get_outcome_ratios(history_data, 'T'))

    # 3. 近期趨勢（最近10局）
    features.append(get_recent_trend(history_data, 10))

    # 4. 當前連勝長度
    features.append(get_streak_length(history_data))

    # 5. 距離上次出現各結果的局數（標準化）
    total_games = len(history_data)
    features.append(get_gap_since_last(history_data, 'B') /
                    (total_games if total_games > 0 else 1.0))
    features.append(get_gap_since_last(history_data, 'P') /
                    (total_games if total_games > 0 else 1.0))
    features.append(get_gap_since_last(history_data, 'T') /
                    (total_games if total_games > 0 else 1.0))

    # 6. 最近5局的B/P比例
    recent_5 = history_data[-5:] if len(history_data) >= 5 else history_data
    features.append(get_outcome_ratios(recent_5, 'B'))
    features.append(get_outcome_ratios(recent_5, 'P'))

    # 7. 新增百家樂俚語模式特徵 (使用更長的回溯窗口確保能識別模式)
    # 考慮最近LOOK_BACK局的歷史來判斷這些模式，以提供足夠上下文
    long_history_for_patterns = history_data[-LOOK_BACK:]
    features.append(is_pattern_long_run(
        long_history_for_patterns, min_len=5))  # 長龍
    features.append(is_pattern_single_jump(
        long_history_for_patterns, min_len=4))  # 單跳
    features.append(is_pattern_double_jump(
        long_history_for_patterns, min_len=6))  # 雙跳 (BBPPBB)
    features.append(is_pattern_two_in_a_row_one_off(
        long_history_for_patterns, min_len=6))  # 一房兩廳 (BBPBBP)

    # 8. 新增虛擬路單特徵
    virtual_road_features = get_virtual_road_features(
        long_history_for_patterns)
    features.append(virtual_road_features['big_road_cols'])
    features.append(virtual_road_features['last_big_road_length'])
    features.append(virtual_road_features['big_road_alternating_pattern'])
    features.append(virtual_road_features['big_road_double_pattern'])
    features.append(virtual_road_features['big_road_flat_line_pattern'])
    features.append(virtual_road_features['big_road_straight_line_pattern'])

    return np.array(features)


def prepare_training_data(history_data, look_back=LOOK_BACK):
    """準備訓練 LightGBM 和 XGBoost 的數據。"""
    X_features = []
    y = []

    local_label_encoder = LabelEncoder()
    local_label_encoder.fit(['B', 'P', 'T'])

    min_required_history = look_back + 1  # 訓練模型至少需要的歷史數據量
    if len(history_data) < min_required_history:
        print(
            f"訓練歷史數據不足 ({len(history_data)} 筆紀錄)。最低要求: {min_required_history}。")
        return np.array([]), np.array([]), local_label_encoder

    for i in range(len(history_data) - look_back):
        current_context = history_data[:i + look_back]
        features = prepare_simplified_features(current_context, look_back)

        if features is not None:
            X_features.append(features)
            y.append(local_label_encoder.transform(
                [history_data[i + look_back]])[0])

    if not X_features:
        return np.array([]), np.array([]), local_label_encoder

    y_labels = np.array(y)

    return np.array(X_features), y_labels, local_label_encoder

# --- 模型架構定義 (只保留XGBoost和LightGBM) ---


def build_lightgbm_model(num_features, num_classes):
    """建立 LightGBM 模型。"""
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        n_estimators=200,  # 增加估計器數量
        learning_rate=0.05,  # 降低學習率
        max_depth=7,  # 增加樹的深度
        subsample=0.7,  # 調整子樣本比例
        colsample_bytree=0.7,  # 調整列採樣比例
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    return model


def build_xgboost_model(num_features, num_classes):
    """建立 XGBoost 模型。"""
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=200,  # 增加估計器數量
        learning_rate=0.05,  # 降低學習率
        max_depth=7,  # 增加樹的深度
        subsample=0.7,  # 調整子樣本比例
        colsample_bytree=0.7,  # 調整列採樣比例
        random_state=42
    )
    return model

# --- 模型訓練和保存 ---


def train_and_save_models(history_data):
    """訓練並保存 LightGBM 和 XGBoost 模型。"""
    global ml_model_lightgbm, ml_model_xgboost, ml_label_encoder, feature_scaler

    min_required_history = LOOK_BACK + 1  # 訓練至少需要 LOOK_BACK + 1 筆記錄
    if len(history_data) < min_required_history:
        return False, f"訓練數據不足 ({len(history_data)} 筆紀錄)。最低要求: {min_required_history}。"

    X_features, y_labels, current_label_encoder = prepare_training_data(
        history_data, LOOK_BACK)

    if len(X_features) == 0 or len(y_labels) == 0: # 確保有足夠的數據進行訓練
        return False, "訓練數據不足，無法有效訓練。"

    ml_label_encoder = current_label_encoder

    # 數據標準化：訓練時重新 fit_transform
    feature_scaler = StandardScaler()
    X_features_scaled = feature_scaler.fit_transform(X_features)

    # 計算類別權重
    unique_y_labels_encoded = np.unique(y_labels) # 獲取 y_labels 中實際存在的唯一編碼標籤
    all_ml_classes_encoded = ml_label_encoder.transform(ml_label_encoder.classes_) # 所有可能的編碼標籤 (0, 1, 2)
    
    # 僅對 y_labels 中存在的類別計算權重
    if len(unique_y_labels_encoded) > 0:
        raw_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_y_labels_encoded, # 傳入 y_labels 中實際存在的類別
            y=y_labels
        )
        raw_class_weight_dict = dict(zip(unique_y_labels_encoded, raw_class_weights))
    else:
        raw_class_weight_dict = {} # 如果 y_labels 為空，則無法計算權重

    # 為所有可能的類別創建完整的權重字典，如果某類別未在 y_labels 中出現，則其權重設為 1.0 (或您希望的預設值)
    class_weight_dict = {}
    for encoded_class in all_ml_classes_encoded:
        class_weight_dict[encoded_class] = raw_class_weight_dict.get(encoded_class, 1.0) # 預設權重為 1.0
        
    sample_weights_array = np.array([class_weight_dict[label] for label in y_labels])

    num_classes = len(ml_label_encoder.classes_)

    # --- 訓練和保存 LightGBM 模型 ---
    print("開始訓練 LightGBM 模型...")
    try:
        lgb_model = build_lightgbm_model(
            X_features_scaled.shape[1], num_classes)
        lgb_model.fit(X_features_scaled, y_labels,
                      sample_weight=sample_weights_array)
        lgb_model.booster_.save_model(MODEL_FILE_LIGHTGBM)
        ml_model_lightgbm = lgb_model
        print("LightGBM 模型已成功訓練並保存。")
    except Exception as e:
        print(f"LightGBM 模型訓練失敗: {e}")
        ml_model_lightgbm = None

    # --- 訓練和保存 XGBoost 模型 ---
    print("開始訓練 XGBoost 模型...")
    try:
        xgb_model = build_xgboost_model(
            X_features_scaled.shape[1], num_classes)
        xgb_model.fit(X_features_scaled, y_labels,
                      sample_weight=sample_weights_array)
        xgb_model.save_model(MODEL_FILE_XGBOOST)
        ml_model_xgboost = xgb_model
        print("XGBoost 模型已成功訓練並保存。")
    except Exception as e:
        print(f"XGBoost 模型訓練失敗: {e}")
        ml_model_xgboost = None

    # 保存 LabelEncoder 和 Scaler
    try:
        joblib.dump(current_label_encoder, MODEL_FILE_ENCODER)
        joblib.dump(feature_scaler, MODEL_FILE_SCALER)
        print("標籤編碼器和特徵縮放器已成功保存。")
    except Exception as e:
        print(f"組件保存失敗: {e}")

    # 更新馬可夫鏈模型
    update_markov_chain(history_data)
    print("馬可夫鏈模型已更新。")

    if ml_model_lightgbm is not None or ml_model_xgboost is not None:
        return True, "模型已成功訓練並載入。"
    else:
        return False, "模型訓練失敗，沒有模型成功載入。"

# --- 模型載入 ---
def load_ml_models():
    """載入模型和組件。"""
    global ml_model_lightgbm, ml_model_xgboost, ml_label_encoder, feature_scaler

    # 載入 LabelEncoder
    try:
        if os.path.exists(MODEL_FILE_ENCODER):
            ml_label_encoder = joblib.load(MODEL_FILE_ENCODER)
            print("標籤編碼器已成功載入。")
        else:
            print("未找到標籤編碼器檔案。初始化一個。")
            ml_label_encoder = LabelEncoder()
            ml_label_encoder.fit(['B', 'P', 'T'])
    except Exception as e:
        print(f"載入標籤編碼器失敗: {e}")
        ml_label_encoder = LabelEncoder()
        ml_label_encoder.fit(['B', 'P', 'T'])

    # 載入特徵縮放器
    try:
        if os.path.exists(MODEL_FILE_SCALER):
            feature_scaler = joblib.load(MODEL_FILE_SCALER)
            print("特徵縮放器已成功載入。")
        else:
            print("未找到特徵縮放器檔案。初始化一個未訓練的 Standard Scaler。")
            try:
                # 使用一個dummy數據進行fit，使其內部狀態得以初始化
                dummy_history = ['B'] * LOOK_BACK
                dummy_features = prepare_simplified_features(dummy_history)
                if dummy_features is not None:
                    feature_scaler = StandardScaler()  # 確保是一個新的實例
                    # 為 feature_scaler.fit 準備一個預期形狀的數據，即使是零
                    feature_scaler.fit(np.zeros((1, dummy_features.shape[0])))
                    print("未訓練的特徵縮放器已初始化並執行假 fit。")
                else:
                    print("無法生成 dummy features，未訓練的特徵縮放器無法執行假 fit。")
            except Exception as fit_e:
                print(f"假 fit 失敗: {fit_e}")

    except Exception as e:
        print(f"載入特徵縮放器失敗: {e}")
        feature_scaler = StandardScaler()

    # 載入 LightGBM 模型
    try:
        if os.path.exists(MODEL_FILE_LIGHTGBM):
            ml_model_lightgbm = lgb.Booster(model_file=MODEL_FILE_LIGHTGBM)
            print("LightGBM 模型已成功載入。")
        else:
            ml_model_lightgbm = None
    except Exception as e:
        print(f"載入 LightGBM 模型失敗: {e}")
        ml_model_lightgbm = None

    # 載入 XGBoost 模型
    try:
        if os.path.exists(MODEL_FILE_XGBOOST):
            ml_model_xgboost = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='mlogloss')
            ml_model_xgboost.load_model(MODEL_FILE_XGBOOST)
            print("XGBoost 模型已成功載入。")
        else:
            ml_model_xgboost = None
    except Exception as e:
        print(f"載入 XGBoost 模型失敗: {e}")
        ml_model_xgboost = None

    return ml_model_lightgbm is not None or ml_model_xgboost is not None

# 在應用程式啟動時載入模型的替代方案
# 這種方式會在 gunicorn worker 啟動時各自執行一次
def initialize_models():
    """在每個 Gunicorn worker 啟動時調用，用於模型初始化。"""
    global ml_label_encoder, feature_scaler # 確保可以修改全局變量
    print("Initializing ML models for worker...")
    if os.path.exists(MODEL_FILE_ENCODER): # 確保至少 LabelEncoder 存在
        load_ml_models()
        print("現有模型已成功載入。")
    else:
        print("未找到任何模型檔案。應用程式將等待歷史數據輸入後進行訓練。")
        # 即使沒有模型文件，也要確保 ml_label_encoder 和 feature_scaler 被初始化
        ml_label_encoder = LabelEncoder()
        ml_label_encoder.fit(['B', 'P', 'T'])
        feature_scaler = StandardScaler()
        # 嘗試對 feature_scaler 進行一次假 fit
        try:
            dummy_history = ['B'] * LOOK_BACK
            dummy_features = prepare_simplified_features(dummy_history)
            if dummy_features is not None:
                feature_scaler = StandardScaler()  # 確保是一個新的實例
                feature_scaler.fit(np.zeros((1, dummy_features.shape[0])))
                print("未訓練的特徵縮放器已初始化並執行假 fit。")
            else:
                print("無法生成 dummy features，未訓練的特徵縮放器無法執行假 fit。")
        except Exception as fit_e:
            print(f"假 fit 失敗: {fit_e}")


# === 為了兼容 Codesandbox 環境，我們將初始化邏輯在 app 實例創建後立即執行一次 ===
initialize_models() # 在應用程式啟動時執行一次模型初始化

# --- 預測邏輯 ---
def predict_next_outcome(history_data):
    """使用多個模型預測下一個結果，並選擇最佳結果。"""
    if ml_label_encoder is None:
        print("Warning: ml_label_encoder is None. Returning fallback prediction.")
        default_probabilities = {'B': 1/3, 'P': 1/3, 'T': 1/3}
        return {
            "prediction": "觀望",
            "probabilities": default_probabilities,
            "confidence": 1/3,
            "source": "fallback"
        }

    min_predict_history_tree_models = LOOK_BACK  # 樹模型預測需要足夠數據來提取俚語特徵
    min_predict_history_markov = MARKOV_LOOK_BACK  # 馬可夫鏈所需的最低歷史數據

    results = {}  # 儲存各模型的預測結果和信心度

    # --- 馬可夫鏈預測 (始終最快，作為 fallback 或補充) ---
    if len(history_data) >= min_predict_history_markov:
        markov_probs = predict_markov_chain(history_data)
        markov_pred_outcome = max(markov_probs, key=markov_probs.get)
        markov_confidence = markov_probs[markov_pred_outcome]
        results['markov'] = {
            "prediction": markov_pred_outcome,
            "probabilities": markov_probs,
            "confidence": markov_confidence,
            "source": "markov"
        }

    # --- 簡化特徵提取 (適用於 LightGBM 和 XGBoost) ---
    features_scaled = None
    if len(history_data) >= min_predict_history_tree_models:
        try:
            features = prepare_simplified_features(history_data)
            # 檢查 feature_scaler 是否已經訓練過 (即具有 n_features_in_ 屬性)
            if features is not None and feature_scaler is not None and \
               hasattr(feature_scaler, 'n_features_in_') and feature_scaler.n_features_in_ is not None:
                features_scaled = feature_scaler.transform([features])
            else:
                print("Warning: 特徵準備失敗或縮放器未訓練，跳過樹模型預測。")
        except Exception as e:
            print(f"特徵準備或縮放失敗: {e}")
    else:
        print(
            f"歷史數據不足 ({len(history_data)} 筆) 以準備樹模型特徵。最低要求: {min_predict_history_tree_models}。")

    # --- LightGBM 預測 ---
    if ml_model_lightgbm is not None and features_scaled is not None:
        try:
            # 修正: 使用 Booster 預測並轉換為概率
            lgb_raw_pred = ml_model_lightgbm.predict(features_scaled)
            # 將 raw score 轉換為概率 (softmax)
            lgb_probs = np.exp(lgb_raw_pred) / np.sum(np.exp(lgb_raw_pred), axis=1, keepdims=True)
            lgb_probs = lgb_probs[0]  # 取第一個樣本的預測
            
            lgb_predicted_label_index = np.argmax(lgb_probs)
            lgb_predicted_outcome = ml_label_encoder.inverse_transform([lgb_predicted_label_index])[0]

            lgb_outcome_probabilities = {
                ml_label_encoder.inverse_transform([i])[0]: float(lgb_probs[i])
                for i in range(len(lgb_probs))
            }
            results['lightgbm'] = {
                "prediction": lgb_predicted_outcome,
                "probabilities": lgb_outcome_probabilities,
                "confidence": float(lgb_probs[lgb_predicted_label_index]),
                "source": "lightgbm"
            }
        except Exception as e:
            print(f"LightGBM 預測錯誤: {e}")

    # --- XGBoost 預測 ---
    if ml_model_xgboost is not None and features_scaled is not None:
        try:
            xgb_probs = ml_model_xgboost.predict_proba(features_scaled)[0]
            xgb_predicted_label_index = np.argmax(xgb_probs)
            xgb_predicted_outcome = ml_label_encoder.inverse_transform(
                [xgb_predicted_label_index])[0]

            xgb_outcome_probabilities = {
                ml_label_encoder.inverse_transform([i])[0]: float(xgb_probs[i])
                for i in range(len(xgb_probs))
            }
            results['xgboost'] = {
                "prediction": xgb_predicted_outcome,
                "probabilities": xgb_outcome_probabilities,
                "confidence": float(xgb_probs[xgb_predicted_label_index]),
                "source": "xgboost"
            }
        except Exception as e:
            print(f"XGBoost 預測錯誤: {e}")

    # --- 模型選擇邏輯 ---
    best_result = None

    # 優先選擇信心度最高的樹模型 (XGBoost 或 LightGBM)
    candidate_tree_models = []
    if 'lightgbm' in results and results['lightgbm']['confidence'] > 0.35:
        candidate_tree_models.append(results['lightgbm'])
    if 'xgboost' in results and results['xgboost']['confidence'] > 0.35:
        candidate_tree_models.append(results['xgboost'])

    if candidate_tree_models:
        best_result = max(candidate_tree_models, key=lambda x: x['confidence'])

    # 如果樹模型無有效結果或信心度不足，退回馬可夫鏈
    # 閾值調整為0.4
    if best_result is None or (best_result and best_result['confidence'] < 0.4):
        if 'markov' in results:
            best_result = results['markov']
        elif not results:  # 如果沒有任何模型有結果 (例如歷史數據不足，或模型未載入)
            default_probabilities = {'B': 1/3, 'P': 1/3, 'T': 1/3}
            return {
                "prediction": "觀望",
                "probabilities": default_probabilities,
                "confidence": 1/3,
                "source": "fallback"
            }
        else:  # 如果有其他模型結果但信心度太低
            best_result = results[list(results.keys())[0]]  # 隨便取一個作為基底
            default_probabilities = {'B': 1/3, 'P': 1/3, 'T': 1/3}  # 強制使用預設機率
            return {
                "prediction": "觀望",
                "probabilities": default_probabilities,
                "confidence": 1/3,
                "source": "fallback"
            }

    # 最終判斷是否觀望，並返回完整的預測結果
    # 如果最佳預測是 "觀望" 或者信心度仍然很低，則強制為觀望
    if best_result['prediction'] == "觀望" or best_result['confidence'] < 0.4:
        return {
            "prediction": "觀望",
            "probabilities": best_result['probabilities'],
            "confidence": best_result['confidence'],
            "source": best_result['source']
        }

    return best_result


# --- Flask 路由 ---
@app.route('/')
def home():
    """根路徑，提供API的基本資訊和使用說明"""
    return jsonify({
        "message": "Baccarat Prediction API is running",
        "endpoints": {
            "GET /": "This homepage",
            "GET /status": "Check model status",
            "POST /predict": "Predict next outcome (requires JSON with 'history')",
            "POST /train": "Train model with new data (requires JSON with 'history')",
            "POST /recommendation": "Get betting recommendation (requires JSON with 'history')",
            "GET /api/history": "Get current game history",
            "POST /api/history": "Save game history",
            "DELETE /api/history": "Clear game history",
            "GET /stats": "Get quick statistics"
        },
        "usage_example": {
            "predict": {"history": ["B", "P", "B", "T", "P"]}
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "請提供 JSON 數據"}), 400

        history = data.get('history', [])
        history = [h.upper() for h in history]

        if not history:
            return jsonify({"error": "請提供歷史數據。"}), 400

        prediction_result = predict_next_outcome(history)

        if isinstance(prediction_result, str):  # 如果 predict_next_outcome 返回的是錯誤訊息字串
            return jsonify({"error": prediction_result}), 400

        return jsonify(prediction_result)
    except Exception as e:
        print(f"predict 路由發生錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500


@app.route('/train', methods=['POST'])
def train_model_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "請提供 JSON 數據"}), 400

        history = data.get('history', [])
        history = [h.upper() for h in history]

        if not history:
            return jsonify({"error": "請提供歷史數據。"}), 400

        success, message = train_and_save_models(history)
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 500
    except Exception as e:
        print(f"train 路由發生錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

# 歷史記錄 API 路由
@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
def handle_history_api():
    global game_history_backend, ai_prediction_outcomes_backend # 管理 ai_prediction_outcomes_backend 的更新
    
    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({"error": "請提供 JSON 數據"}), 400
            
            game_history_backend = data.get('history', []) # 只更新遊戲歷史
            # ai_accuracy_history 不再從前端接收，由後端自己的 /api/record_ai_performance 管理

            min_train_history = LOOK_BACK + 1
            if len(game_history_backend) >= min_train_history:
                train_and_save_models(game_history_backend)
                print(
                    f"Received history from frontend: {len(game_history_backend)} records. Models re-trained.")
            else:
                update_markov_chain(game_history_backend)
                print(
                    f"Received history from frontend: {len(game_history_backend)} records. Not enough for tree models, Markov chain updated.")

            return jsonify({"message": "History saved and models updated successfully", "records": len(game_history_backend)}), 200
        elif request.method == 'GET':
            # 返回 JSON 物件
            return jsonify({"history": game_history_backend, "ai_accuracy_history": ai_prediction_outcomes_backend}), 200
        elif request.method == 'DELETE':
            game_history_backend = []
            ai_prediction_outcomes_backend = [] # 清除 AI 準確率歷史
            initialize_models()  # 清除歷史後，重新初始化模型
            print("History cleared from backend, models re-initialized.")
            return jsonify({"message": "History and AI accuracy history cleared and models re-initialized successfully"}), 200
        return jsonify({"error": "Method not allowed"}), 405
    except Exception as e:
        print(f"/api/history 路由發生錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

# --- 新增 AI 預測表現記錄端點 ---
@app.route('/api/record_ai_performance', methods=['POST', 'DELETE'])
def record_ai_performance():
    global ai_prediction_outcomes_backend

    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({"error": "請提供 JSON 數據"}), 400
            
            actual_result = data.get('actual_result')
            ai_predicted_outcome = data.get('ai_predicted_outcome')
            
            if not actual_result or not ai_predicted_outcome:
                return jsonify({"error": "請提供 actual_result 和 ai_predicted_outcome"}), 400
            
            # 只在 AI 預測是 B 或 P 時才記錄準確性
            if ai_predicted_outcome in ['B', 'P']:
                is_correct = (actual_result == ai_predicted_outcome)
                ai_prediction_outcomes_backend.append(is_correct)
                print(f"Recorded AI performance: Actual={actual_result}, Predicted={ai_predicted_outcome}, Correct={is_correct}")
                return jsonify({"message": "AI performance recorded", "is_correct": is_correct}), 200
            else:
                print(f"Skipping AI performance record: AI predicted {ai_predicted_outcome}")
                return jsonify({"message": "AI prediction not for B/P, skipping record"}), 200
        
        elif request.method == 'DELETE': # 處理撤銷操作
            if ai_prediction_outcomes_backend:
                ai_prediction_outcomes_backend.pop()
                print("Last AI performance record undone.")
                return jsonify({"message": "Last AI performance record undone"}), 200
            else:
                return jsonify({"message": "No AI performance record to undo"}), 404
        
        return jsonify({"error": "Method not allowed"}), 405
    except Exception as e:
        print(f"/api/record_ai_performance 路由發生錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500


@app.route('/status', methods=['GET'])
def status():
    try:
        model_xgboost_loaded = ml_model_xgboost is not None
        model_lightgbm_loaded = ml_model_lightgbm is not None 
        encoder_loaded = ml_label_encoder is not None
        scaler_loaded = feature_scaler is not None and hasattr(
            feature_scaler, 'n_features_in_') and feature_scaler.n_features_in_ is not None

        return jsonify({
            "xgboost_model_loaded": model_xgboost_loaded,
            "lightgbm_model_loaded": model_lightgbm_loaded,
            "encoder_loaded": encoder_loaded,
            "scaler_loaded": scaler_loaded,
            "xgboost_model_file_exists": os.path.exists(MODEL_FILE_XGBOOST),
            "lightgbm_model_file_exists": os.path.exists(MODEL_FILE_LIGHTGBM),
            "encoder_file_exists": os.path.exists(MODEL_FILE_ENCODER),
            "scaler_file_exists": os.path.exists(MODEL_FILE_SCALER),
            "look_back_length": LOOK_BACK,
            "current_history_length": len(game_history_backend),
            "ai_performance_records": len(ai_prediction_outcomes_backend) # 新增 AI 表現記錄數
        })
    except Exception as e:
        print(f"status 路由發生錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

# Constants for room detection in backend
ROOM_ACCURACY_WINDOW = 15  # Calculate AI accuracy over last X games
ROOM_CHANGE_THRESHOLD = 0.40  # Below this, suggest changing room
ROOM_WARNING_THRESHOLD = 0.55  # Below this, warn about performance


@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    """提供下注建議，考慮風險和回報，並增加房間偵測。"""
    global ai_prediction_outcomes_backend # 確保能訪問全局變量

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "請提供 JSON 數據"}), 400

        history = data.get('history', [])
        history = [h.upper() for h in history]
        
        if not history:
            return jsonify({"error": "請提供歷史數據。"}), 400
        
        prediction_result = predict_next_outcome(history)
        
        if isinstance(prediction_result, str):  # 如果 predict_next_outcome 返回的是錯誤訊息字串
            return jsonify({"error": prediction_result}), 400
        
        prediction = prediction_result["prediction"]
        probabilities = prediction_result["probabilities"]
        confidence = prediction_result["confidence"]
        source_model = prediction_result.get("source", "unknown")

        recommendation = "建議觀望，市場不明確"
        bet_amount = "不下注"

        if prediction != "觀望" and confidence >= 0.4:
            # 找到機率最高的結果
            highest_prob_outcome = max(probabilities, key=probabilities.get)
            
            # 判斷下注建議等級
            if confidence < 0.6:
                recommendation = f"輕度看好 {highest_prob_outcome} (來源: {source_model})，但風險較高"
                bet_amount = "小注"
            elif confidence < 0.8:
                recommendation = f"中度看好 {highest_prob_outcome} (來源: {source_model})"
                bet_amount = "中等注碼"
            else:
                recommendation = f"強烈看好 {highest_prob_outcome} (來源: {source_model})"
                bet_amount = "大注"

        # --- 新增房間偵測邏輯 (直接使用後端維護的 ai_prediction_outcomes_backend) ---
        room_accuracy_percentage = 0.0
        room_status_text = "AI 表現良好"
        room_status_class = "room-good"

        if len(ai_prediction_outcomes_backend) >= ROOM_ACCURACY_WINDOW:
            recent_outcomes = ai_prediction_outcomes_backend[-ROOM_ACCURACY_WINDOW:]
            correct_count = sum(1 for x in recent_outcomes if x is True) # Count True values
            room_accuracy_percentage = correct_count / len(recent_outcomes)

            if room_accuracy_percentage < ROOM_CHANGE_THRESHOLD:
                room_status_text = "建議換房，AI預測表現不佳。"
                room_status_class = "room-bad"
            elif room_accuracy_percentage < ROOM_WARNING_THRESHOLD:
                room_status_text = "AI 表現中等，請持續觀察。"
                room_status_class = "room-warning"
            else:
                room_status_text = "AI 表現良好"
                room_status_class = "room-good"
        elif len(ai_prediction_outcomes_backend) > 0: # Not enough for full window, but some data
            correct_count = sum(1 for x in ai_prediction_outcomes_backend if x is True)
            room_accuracy_percentage = correct_count / len(ai_prediction_outcomes_backend)
            if room_accuracy_percentage < ROOM_CHANGE_THRESHOLD:
                room_status_text = "AI 表現不佳。"
                room_status_class = "room-bad"
            else:
                room_status_text = "AI 表現良好 (數據較少)"
                room_status_class = "room-good"
        else:
            room_status_text = "AI 表現良好 (無數據)"
            room_status_class = "room-good"


        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": confidence,
            "recommendation": recommendation,
            "bet_amount": bet_amount,
            "source_model": source_model,
            "room_accuracy_percentage": round(room_accuracy_percentage * 100, 1), # Return as percentage
            "room_status_text": room_status_text,
            "room_status_class": room_status_class,
            "room_accuracy_window": ROOM_ACCURACY_WINDOW # Pass window size for frontend display
        })
    except Exception as e:
        print(f"recommendation 路由發生錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """快速統計接口"""
    try:
        total = len(game_history_backend)
        banker_count = game_history_backend.count('B')
        player_count = game_history_backend.count('P')
        tie_count = game_history_backend.count('T')
        
        return jsonify({
            "total_games": total,
            "banker_wins": banker_count,
            "player_wins": player_count,
            "ties": tie_count,
            "banker_ratio": round(banker_count / total, 3) if total > 0 else 0,
            "player_ratio": round(player_count / total, 3) if total > 0 else 0,
            "tie_ratio": round(tie_count / total, 3) if total > 0 else 0,
        })
    except Exception as e:
        print(f"stats 路由發生錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500


@app.route('/api/bigroad', methods=['POST'])
def get_bigroad():
    """從前端獲取歷史數據並返回大路地圖。"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "請提供 JSON 數據"}), 400
        history = data.get('history', [])
        history = [h.upper() for h in history]

        if not history:
            return jsonify({"error": "請提供歷史數據以生成大路圖。"}), 400

        bigroad_map = _get_big_road_map(history)
        # 返回一個 JSON 格式的二維列表
        return jsonify({"big_road_map": bigroad_map}), 200
    except Exception as e:
        print(f"/api/bigroad 路由發生錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
