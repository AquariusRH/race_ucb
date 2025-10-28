import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time

# ==================== 你的 API 函數（優化版）====================
@st.cache_data(ttl=3600)  # 每小時更新一次基本資料
def get_race_basic_data(Date, place):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "raceMeetings",
        "variables": {"date": str(Date), "venueCode": place},
        "query": """
        query raceMeetings($date: String, $venueCode: String) {
          raceMeetings(date: $date, venueCode: $venueCode) {
            races {
              no
              postTime
              runners {
                id
                no
                standbyNo
                name_ch
                jockey { name_ch }
                trainer { name_ch }
                last6run
              }
            }
          }
        }
        """
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code != 200: return None, None
        data = r.json().get('data', {}).get('raceMeetings', [])
        if not data: return None, None

        race_dict = {}
        post_time_dict = {}
        for meeting in data:
            for race in meeting.get('races', []):
                race_no = race['no']
                post_time = race.get('postTime')
                if not post_time: continue
                post_time_dt = datetime.fromisoformat(post_time.replace('Z', '+00:00')).astimezone()
                post_time_dict[race_no] = post_time_dt

                race_dict[race_no] = {
                    "馬名": [], "騎師": [], "練馬師": [], "最近賽績": [], "馬號": []
                }
                for runner in race.get('runners', []):
                    if runner.get('standbyNo'): continue
                    no = runner.get('no')
                    race_dict[race_no]["馬號"].append(no)
                    race_dict[race_no]["馬名"].append(runner.get('name_ch', '未知'))
                    race_dict[race_no]["騎師"].append(runner.get('jockey', {}).get('name_ch', '未知'))
                    race_dict[race_no]["練馬師"].append(runner.get('trainer', {}).get('name_ch', '未知'))
                    race_dict[race_no]["最近賽績"].append(runner.get('last6run', '-'))
        return race_dict, post_time_dict
    except:
        return None, None

# ==================== 其他 API（簡化）====================
def get_odds_data(Date, place, race_no):
    # ...（你的 WIN odds 函數）
    pass

def get_investment_data(Date, place, race_no):
    # ...（你的 investment 函數）
    pass

# ==================== Streamlit App ====================
st.set_page_config(page_title="UCB 最終預測表格", layout="wide")
st.title("UCB 最終預測表格（直至開跑）")

# --- 輸入 ---
col1, col2 = st.columns([1, 2])
with col1:
    Date = st.date_input("日期", datetime.today())
    place = st.selectbox("場地", ["ST", "HV"])
    race_no = st.number_input("賽事", 1, 12, 1, step=1)
    top_k = st.number_input("每次選幾匹", 2, 6, 4)

# --- 抓基本資料 ---
with st.spinner("正在載入賽事基本資料..."):
    race_dict, post_time_dict = get_race_basic_data(Date, place)

if not race_dict or race_no not in race_dict:
    st.error("無法取得賽事資料，請檢查日期/場地/賽事編號")
    st.stop()

# 取得開賽時間
post_time = post_time_dict.get(race_no)
if not post_time:
    st.error("無開賽時間")
    st.stop()

# --- 狀態 ---
if 'state' not in st.session_state:
    st.session_state.state = {
        't': 0,
        'selected_count': {},
        'momentum_hist': {},
        'final_table': None,
        'locked': False,
        'start_time': None
    }

state = st.session_state.state

# --- 開始監測 ---
if st.button("開始監測本場", type="primary"):
    state.update({
        't': 0,
        'selected_count': {i+1: 0 for i in range(len(race_dict[race_no]['馬號']))},
        'momentum_hist': {i+1: [] for i in range(len(race_dict[race_no]['馬號']))},
        'final_table': None,
        'locked': False,
        'start_time': datetime.now()
    })
    st.rerun()

if not state['start_time']:
    st.stop()

# --- 倒數計時 ---
now = datetime.now()
time_to_race = post_time - now
mins, secs = divmod(max(int(time_to_race.total_seconds()), 0), 60)

st.markdown(f"### 距離開賽：**{mins} 分 {secs} 秒**")

# --- 鎖定邏輯：開賽前 1 分鐘 ---
if time_to_race.total_seconds() <= 60 and not state['locked']:
    state['locked'] = True
    # 鎖定最終表格
    if state['final_table'] is not None:
        st.success("開賽前 1 分鐘：最終預測鎖定！")

# --- 主循環：每 30 秒更新 ---
if not state['locked'] and state['t'] == 0 or (now - state['start_time']).total_seconds() >= state['t'] * 30:
    state['t'] += 1
    t = state['t']

    # 抓即時資料
    odds_list = get_odds_data(Date, place, race_no) or [999]*len(race_dict[race_no]['馬號'])
    total_inv = get_investment_data(Date, place, race_no) or 0

    win_odds = np.array([o if o < 999 else 999 for o in odds_list])
    n_horses = len(win_odds)
    horses = list(range(1, n_horses + 1))

    # 隱含投注量
    inv_per_horse = total_inv / win_odds if total_inv > 0 else np.zeros(n_horses)

    # 正規化動量
    momentum = {}
    inv_mean = np.mean(inv_per_horse) if np.mean(inv_per_horse) > 0 else 1
    for h in horses:
        idx = h - 1
        state['momentum_hist'][h].append(inv_per_horse[idx])
        hist = state['momentum_hist'][h]
        raw_mom = hist[-1] - hist[-2] if len(hist) >= 2 else 0
        momentum[h] = raw_mom / (inv_mean + 1e-6)

    # UCB 計算（防熱門）
    ucb_values = {}
    for h in horses:
        n = max(state['selected_count'][h], 1)
        exploration = 2.0 * np.sqrt(np.log(t) / n)
        value_bonus = 1.0 / (win_odds[h-1] ** 0.5)
        avg_sel = np.mean(list(state['selected_count'].values()))
        penalty = -0.3 * (state['selected_count'][h] - avg_sel)**2
        ucb_values[h] = momentum[h]*2.0 + exploration + value_bonus + penalty

    # 選 Top K
    sorted_h = sorted(ucb_values.items(), key=lambda x: x[1], reverse=True)
    topk = [h for h, _ in sorted_h[:top_k]]
    for h in topk:
        state['selected_count'][h] += 1

    # --- 建立最終表格 ---
    table_data = []
    for h in horses:
        idx = h - 1
        table_data.append({
            '馬號': race_dict[race_no]['馬號'][idx],
            '馬名': race_dict[race_no]['馬名'][idx],
            '騎師': race_dict[race_no]['騎師'][idx],
            '練馬師': race_dict[race_no]['練馬師'][idx],
            '最近賽績': race_dict[race_no]['最近賽績'][idx],
            '賠率': f"{win_odds[idx]:.2f}",
            '動量': f"{momentum[h]:+.3f}",
            '被選次數': state['selected_count'][h],
            'UCB': f"{ucb_values[h]:.3f}",
            '排名': f"Top {topk.index(h)+1}" if h in topk else ""
        })
    df = pd.DataFrame(table_data).sort_values('UCB', ascending=False)

    # 存入 state
    state['final_table'] = df
    st.rerun()

# --- 顯示最終表格 ---
if state['final_table'] is not None:
    df = state['final_table'].copy()

    # 高亮 Top 4
    def highlight_top(row):
        if row['排名'].startswith('Top'):
            rank = int(row['排名'].split()[1])
            colors = ["#90EE90", "#FFFFE0", "#FFB6C1", "#87CEEB"]
            return [f'background-color: {colors[rank-1]}' for _ in row]
        return [''] * len(row)

    styled_df = df.style.apply(highlight_top, axis=1).format({
        '賠率': '{:.2f}', '動量': '{:+.3f}', 'UCB': '{:.3f}'
    })

    if state['locked']:
        st.success("最終預測（已鎖定）")
        st.dataframe(styled_df, use_container_width=True)
        top4 = df.head(4)['馬號'].tolist()
        st.info(f"建議投注：PLA {top4} + QIN(前二)")
    else:
        st.info(f"即時預測（更新 {state['t']}）")
        st.dataframe(styled_df, use_container_width=True)

# --- 開賽倒數 ---
if time_to_race.total_seconds() > 0:
    st.markdown(f"**開賽時間：{post_time.strftime('%H:%M:%S')}**")
else:
    st.balloons()
    st.success("比賽已開始！")
