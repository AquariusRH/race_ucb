import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# ==================== HKJC API 函數 ====================
def get_odds_data(Date, place, race_no):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "racing",
        "variables": {"date": str(Date), "venueCode": place, "raceNo": int(race_no), "oddsTypes": ["WIN"]},
        "query": """
        query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
          raceMeetings(date: $date, venueCode: $venueCode) {
            pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
              oddsType
              oddsNodes { combString oddsValue }
            }
          }
        }
        """
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code != 200: return []
        data = r.json().get('data', {}).get('raceMeetings', [])
        if not data: return []
        odds = []
        for meeting in data:
            for pool in meeting.get('pmPools', []):
                if pool.get('oddsType') != 'WIN': continue
                for node in pool.get('oddsNodes', []):
                    val = node.get('oddsValue')
                    if val in ['SCR', '---', None]: val = 999
                    else: val = float(val)
                    odds.append(val)
        return odds
    except: return []

def get_investment_data(Date, place, race_no):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "racing",
        "variables": {"date": str(Date), "venueCode": place, "raceNo": int(race_no), "oddsTypes": ["WIN"]},
        "query": """
        query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
          raceMeetings(date: $date, venueCode: $venueCode) {
            poolInvs: pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) { investment }
          }
        }
        """
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code != 200: return 0
        data = r.json().get('data', {}).get('raceMeetings', [])
        if not data: return 0
        for meeting in data:
            for pool in meeting.get('poolInvs', []):
                if pool.get('investment') is not None:
                    return float(pool.get('investment', 0))
        return 0
    except: return 0

# ==================== Streamlit App ====================
st.set_page_config(page_title="UCB Top4 防熱門版", layout="wide")
st.title("UCB 每次選 4 匹（防熱門版）")

# --- 輸入 ---
col1, col2 = st.columns([1, 2])
with col1:
    Date = st.date_input("日期", datetime.today())
    place = st.selectbox("場地", ["ST", "HV"])
    race_no = st.number_input("賽事", 1, 12, 1)
    top_k = st.number_input("每次選幾匹", 2, 6, 4)
    c_weight = st.slider("探索權重", 0.5, 5.0, 2.0)
    momentum_weight = st.slider("動量權重", 0.0, 5.0, 2.0)
    value_weight = st.slider("價值權重", 0.0, 3.0, 1.0)
    penalty_weight = st.slider("過熱懲罰", 0.0, 1.0, 0.3)

# --- 狀態初始化 ---
if 'state' not in st.session_state:
    st.session_state.state = {
        't': 0,
        'selected_count': {},
        'momentum_hist': {},
        'current_topk': [],
        'all_predictions': [],
        'start_time': None
    }

if st.button("開始監測本場", type="primary"):
    st.session_state.state = {
        't': 0,
        'selected_count': {},
        'momentum_hist': {},
        'current_topk': [],
        'all_predictions': [],
        'start_time': datetime.now()
    }
    st.rerun()

state = st.session_state.state
if not state['start_time']:
    st.stop()

# --- 主循環：每 30 秒更新 ---
now = datetime.now()
elapsed = (now - state['start_time']).total_seconds()
if state['t'] == 0 or elapsed >= state['t'] * 30:
    state['t'] += 1
    t = state['t']

    # 抓資料
    odds_list = get_odds_data(Date, place, race_no)
    total_inv = get_investment_data(Date, place, race_no)

    if len(odds_list) < top_k:
        st.warning("馬匹不足")
        st.stop()

    win_odds = np.array([o if o < 999 else 999 for o in odds_list])
    n_horses = len(win_odds)
    horses = list(range(1, n_horses + 1))

    # 隱含投注量
    inv_per_horse = np.zeros(n_horses)
    if total_inv > 0:
        inv_per_horse = total_inv / win_odds
        inv_per_horse[np.isinf(inv_per_horse)] = 0

    # 初始化
    if not state['selected_count']:
        state['selected_count'] = {h: 0 for h in horses}
        state['momentum_hist'] = {h: [] for h in horses}

    # === 防熱門機制 1：正規化動量 ===
    momentum = {}
    current_inv = [inv_per_horse[h-1] for h in horses]
    inv_mean = np.mean(current_inv) if np.mean(current_inv) > 0 else 1
    for h in horses:
        idx = h - 1
        state['momentum_hist'][h].append(inv_per_horse[idx])
        hist = state['momentum_hist'][h]
        raw_momentum = hist[-1] - hist[-2] if len(hist) >= 2 else 0
        momentum[h] = raw_momentum / (inv_mean + 1e-6)  # 正規化

    # === 防熱門機制 2：價值加分 ===
    value_bonus = {}
    for h in horses:
        odds = win_odds[h-1]
        value_bonus[h] = value_weight / (odds ** 0.5)  # 賠率越高，加分越多

    # === 防熱門機制 3：過熱懲罰 ===
    avg_selected = np.mean(list(state['selected_count'].values()))
    overheat_penalty = {}
    for h in horses:
        diff = state['selected_count'][h] - avg_selected
        overheat_penalty[h] = -penalty_weight * diff ** 2

    # === UCB 計算（防熱門版）===
    ucb_values = {}
    for h in horses:
        n = max(state['selected_count'][h], 1)
        exploration = c_weight * np.sqrt(np.log(t) / n)
        ucb = (
            momentum[h] * momentum_weight +
            exploration +
            value_bonus[h] +
            overheat_penalty[h]
        )
        ucb_values[h] = ucb

    # 每次選 Top K 匹
    sorted_horses = sorted(ucb_values.items(), key=lambda x: x[1], reverse=True)
    current_topk = [h for h, _ in sorted_horses[:top_k]]

    # 這 K 匹都被選中
    for h in current_topk:
        state['selected_count'][h] += 1

    # 記錄
    state['all_predictions'].append({
        't': t,
        'topk': current_topk.copy(),
        'ucb': {h: ucb_values[h] for h in current_topk},
        'odds': {h: win_odds[h-1] for h in current_topk}
    })
    state['current_topk'] = current_topk

    st.rerun()

# --- 即時顯示 ---
st.subheader(f"第 {race_no} 場 - 更新 {state['t']} ({state['t']*30}s)")

# 當前 Top K
cols = st.columns(top_k)
for i, h in enumerate(state['current_topk']):
    with cols[i]:
        st.metric(
            label=f"第 {i+1} 名",
            value=f"馬 {h}",
            delta=f"{win_odds[h-1]:.2f}"
        )

# 完整表格
df = pd.DataFrame([
    {
        '馬號': h,
        '賠率': f"{win_odds[h-1]:.2f}",
        '動量': f"{momentum[h]:+.3f}",
        '價值分': f"{value_bonus[h]:.2f}",
        '懲罰': f"{overheat_penalty[h]:.2f}",
        '被選': state['selected_count'][h],
        'UCB': f"{ucb_values[h]:.3f}",
        '排名': f"Top {state['current_topk'].index(h)+1}" if h in state['current_topk'] else ""
    }
    for h in horses
]).set_index('馬號')

def highlight(row):
    if row.name in state['current_topk']:
        rank = state['current_topk'].index(row.name) + 1
        colors = ["#90EE90", "#FFFFE0", "#FFB6C1", "#87CEEB"]
        return [f'background-color: {colors[rank-1]}' for _ in row]
    return [''] * len(row)

st.dataframe(df.style.apply(highlight, axis=1))

# 累積被選次數圖
count_data = [{'馬號': h, '次數': state['selected_count'][h]} for h in horses]
count_df = pd.DataFrame(count_data).sort_values('次數', ascending=False)
st.bar_chart(count_df.set_index('馬號'))

# 最終預測
if state['t'] >= 12:
    final = sorted(state['selected_count'].items(), key=lambda x: x[1], reverse=True)[:4]
    final_horses = [h for h, c in final]
    st.success(f"最終 Top 4：{', '.join([f'馬 {h}' for h in final_horses])}")
    st.info("建議：PLA 4 匹 + QIN(Top2) + TRI(Top3)")

# 趨勢圖
if len(state['all_predictions']) > 1:
    fig = go.Figure()
    for h in horses:
        selected = [1 if h in p['topk'] else 0 for p in state['all_predictions']]
        cum = np.cumsum(selected)
        fig.add_trace(go.Scatter(x=list(range(1, len(cum)+1)), y=cum, name=f"馬 {h}"))
    fig.update_layout(title="每次被選累積", xaxis_title="更新次數")
    st.plotly_chart(fig)
