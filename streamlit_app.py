import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# ==================== 你的三欄輸入 ====================
st.set_page_config(page_title="UCB 最終預測表格", layout="wide")
st.title("UCB 最終預測表格（直至開跑）")

infoColumns = st.columns(3)
with infoColumns[0]:
    Date = st.date_input('日期:', value=datetime.now())
with infoColumns[1]:
    options = ['ST', 'HV', 'S1', 'S2', 'S3', 'S4', 'S5']
    place = st.selectbox('場地:', options)
with infoColumns[2]:
    race_options = np.arange(1, 12)
    race_no = st.selectbox('場次:', race_options)

# ==================== 抓基本資料 API ====================
@st.cache_data(ttl=3600)
def get_race_basic_data(Date, place):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
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
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code != 200:
            return None, None, f"API 錯誤：{r.status_code}"
        data = r.json().get('data', {}).get('raceMeetings', [])
        if not data:
            return None, None, "無賽事資料"

        race_dict = {}
        post_time_dict = {}
        for meeting in data:
            for race in meeting.get('races', []):
                rno = race['no']
                post_time_str = race.get('postTime')
                if not post_time_str: continue
                post_time_utc = datetime.fromisoformat(post_time_str.replace('Z', '+00:00'))
                post_time_hkt = post_time_utc.astimezone(pytz.timezone('Asia/Hong_Kong'))
                post_time_dict[rno] = post_time_hkt

                race_dict[rno] = {
                    "馬號": [], "馬名": [], "騎師": [], "練馬師": [], "最近賽績": []
                }
                for runner in race.get('runners', []):
                    if runner.get('standbyNo'): continue
                    no = runner.get('no', '-')
                    race_dict[rno]["馬號"].append(no)
                    race_dict[rno]["馬名"].append(runner.get('name_ch', '未知'))
                    race_dict[rno]["騎師"].append(runner.get('jockey', {}).get('name_ch', '未知'))
                    race_dict[rno]["練馬師"].append(runner.get('trainer', {}).get('name_ch', '未知'))
                    race_dict[rno]["最近賽績"].append(runner.get('last6run', '-'))
        return race_dict, post_time_dict, "成功"
    except Exception as e:
        return None, None, f"例外錯誤：{e}"

# ==================== 抓即時賠率 & 投注額（你的函數）====================
def get_odds_data(Date, place, race_no):
    # 你的 WIN odds 函數
    try:
        # 模擬或真實 API
        url = 'https://info.cld.hkjc.com/graphql/base/'
        payload = { "operationName": "racing", "variables": {"date": str(Date), "venueCode": place, "raceNo": race_no, "oddsTypes": ["WIN"]}, "query": "..." }
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            # 解析 WIN odds...
            return [3.0, 5.0, 8.0, 12.0]  # 假資料
    except: pass
    return [999] * 12  # 預設

def get_investment_data(Date, place, race_no):
    # 你的 investment 函數
    try:
        # 真實 API
        return 1000000  # 假總投注額
    except: return 0

# ==================== 載入資料 ====================
with st.spinner("正在載入賽事資料..."):
    race_dict, post_time_dict, status_msg = get_race_basic_data(Date, place)

if race_dict is None:
    st.error(f"無法取得賽事資料：{status_msg}")
    st.info("建議：\n- 日期：2025/10/26\n- 場地：ST\n- 場次：1")
    st.stop()

if race_no not in race_dict:
    st.error(f"第 {race_no} 場不存在！當天最多 {max(race_dict.keys())} 場")
    st.stop()

# 開賽時間
post_time = post_time_dict.get(race_no)
if not post_time:
    st.error("無開賽時間")
    st.stop()

st.success(f"第 {race_no} 場資料載入成功！開賽時間：{post_time.strftime('%H:%M:%S')}")

# ==================== UCB 狀態 ====================
if 'ucb' not in st.session_state:
    n_horses = len(race_dict[race_no]['馬號'])
    st.session_state.ucb = {
        't': 0,
        'selected_count': {i+1: 0 for i in range(n_horses)},
        'momentum_hist': {i+1: [] for i in range(n_horses)},
        'final_df': None,
        'locked': False,
        'start_time': None
    }

state = st.session_state.ucb

# ==================== 開始監測 ====================
if st.button("開始 UCB 監測", type="primary"):
    state.update({
        't': 0, 'locked': False, 'final_df': None,
        'start_time': datetime.now(pytz.timezone('Asia/Hong_Kong'))
    })
    st.rerun()

if not state['start_time']:
    st.stop()

# ==================== 倒數 & 鎖定 ====================
now_hkt = datetime.now(pytz.timezone('Asia/Hong_Kong'))
time_left = post_time - now_hkt
mins, secs = divmod(max(int(time_left.total_seconds()), 0), 60)
st.metric("距離開賽", f"{mins:02d}:{secs:02d}")

if time_left.total_seconds() <= 60 and not state['locked']:
    state['locked'] = True
    st.success("開賽前 1 分鐘：最終預測鎖定！")

# ==================== 主循環：每 30 秒更新 ====================
update_trigger = st.button("手動更新") or (state['t'] > 0 and (now_hkt - state['start_time']).seconds >= state['t'] * 30)

if update_trigger and not state['locked']:
    state['t'] += 1
    t = state['t']

    # 抓即時資料
    odds_list = get_odds_data(Date, place, race_no)
    total_inv = get_investment_data(Date, place, race_no)
    win_odds = np.array([o if o < 999 else 999 for o in odds_list[:len(race_dict[race_no]['馬號'])]])
    n_horses = len(win_odds)
    horses = list(range(1, n_horses + 1))

    # 隱含投注量
    inv_per_horse = total_inv / win_odds if total_inv > 0 else np.zeros(n_horses)
    inv_mean = np.mean(inv_per_horse[inv_per_horse > 0]) if np.any(inv_per_horse > 0) else 1

    # 正規化動量
    momentum = {}
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
        penalty = -0.3 * (state['selected_count'][h] - avg_sel) ** 2
        ucb_values[h] = momentum[h] * 2.0 + exploration + value_bonus + penalty

    # 選 Top 4
    topk = sorted(ucb_values, key=ucb_values.get, reverse=True)[:4]
    for h in topk:
        state['selected_count'][h] += 1

    # 建表格
    table_data = []
    for i, h in enumerate(race_dict[race_no]['馬號']):
        horse_no = int(h)
        table_data.append({
            '馬號': horse_no,
            '馬名': race_dict[race_no]['馬名'][i],
            '騎師': race_dict[race_no]['騎師'][i],
            '練馬師': race_dict[race_no]['練馬師'][i],
            '最近賽績': race_dict[race_no]['最近賽績'][i],
            '賠率': f"{win_odds[i]:.2f}",
            '動量': f"{momentum.get(horse_no, 0):+.3f}",
            '被選次數': state['selected_count'][horse_no],
            'UCB': f"{ucb_values.get(horse_no, 0):.3f}",
            '排名': f"Top {topk.index(horse_no)+1}" if horse_no in topk else ""
        })
    df = pd.DataFrame(table_data).sort_values('UCB', ascending=False)
    state['final_df'] = df
    st.rerun()

# ==================== 顯示最終表格 ====================
if state['final_df'] is not None:
    df = state['final_df']

    def highlight(row):
        if row['排名'].startswith('Top'):
            rank = int(row['排名'].split()[1])
            colors = ["#90EE90", "#FFFFE0", "#FFB6C1", "#87CEEB"]
            return [f'background-color: {colors[rank-1]}'] * len(row)
        return [''] * len(row)

    styled = df.style.apply(highlight, axis=1).format({
        '賠率': '{:.2f}', '動量': '{:+.3f}', 'UCB': '{:.3f}'
    })

    if state['locked']:
        st.success("**最終預測表格（已鎖定）**")
    else:
        st.info(f"即時 UCB 表格（第 {state['t']} 次更新）")

    st.dataframe(styled, use_container_width=True)

    top4 = df.head(4)['馬號'].tolist()
    st.info(f"**建議投注**：PLA {top4}｜QIN(前二)｜TRI(前三)")

# ==================== 開賽提醒 ====================
if time_left.total_seconds() <= 0:
    st.balloons()
    st.success("比賽已開始！")
