import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go

# === 你的函數（已優化）===
def get_investment_data(Date, place, race_no, methodlist):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "racing",
        "variables": {"date": str(Date), "venueCode": place, "raceNo": int(race_no), "oddsTypes": methodlist},
        "query": """
        query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
          raceMeetings(date: $date, venueCode: $venueCode) {
            totalInvestment
            poolInvs: pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
              oddsType
              investment
            }
          }
        }
        """
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            data = r.json().get('data', {}).get('raceMeetings', [])
            if not data: return {}
            invs = {m: [] for m in methodlist}
            for meeting in data:
                for pool in meeting.get('poolInvs', []):
                    ot = pool.get('oddsType')
                    if ot in methodlist:
                        invs[ot].append(pool.get('investment', 0))
            return invs
    except: pass
    return {}

def get_odds_data(Date, place, race_no, methodlist):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "racing",
        "variables": {"date": str(Date), "venueCode": place, "raceNo": int(race_no), "oddsTypes": methodlist},
        "query": """
        query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
          raceMeetings(date: $date, venueCode: $venueCode) {
            pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
              oddsType
              oddsNodes {
                combString
                oddsValue
              }
            }
          }
        }
        """
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            data = r.json().get('data', {}).get('raceMeetings', [])
            if not data: return {}
            odds = {m: [] for m in methodlist}
            for meeting in data:
                for pool in meeting.get('pmPools', []):
                    ot = pool.get('oddsType')
                    if ot not in methodlist: continue
                    for node in pool.get('oddsNodes', []):
                        val = node.get('oddsValue')
                        if val in ['SCR', '---', None]: val = np.inf
                        else: val = float(val)
                        if ot in ['QIN','QPL','FCT','TRI','FF']:
                            comb = node.get('combString', '')
                            odds[ot].append((comb, val))
                        else:
                            odds[ot].append(val)
            for ot in ['QIN','QPL','FCT','TRI','FF']:
                odds[ot].sort(key=lambda x: x[0])
            return odds
    except: pass
    return {}

# === Streamlit App ===
st.set_page_config(page_title="UCB 即時賽馬預測", layout="wide")
st.title("UCB + 投注動量 即時預測器")

# === 輸入參數 ===
col1, col2 = st.columns(2)
with col1:
    Date = st.date_input("比賽日期", datetime.today())
    place = st.selectbox("場地", ["ST", "HV"])
    race_no = st.number_input("賽事編號", 1, 12, 1)
with col2:
    methodlist = st.multiselect("投注類型", ["WIN", "PLA"], ["WIN"])
    c_weight = st.slider("探索權重 c", 0.5, 5.0, 2.0)

# === 初始化狀態（每場獨立）===
if 'ucb_state' not in st.session_state:
    st.session_state.ucb_state = {
        't': 0,
        'selected_count': {},
        'momentum_history': {},
        'odds_history': {},
        'investment_history': {},
        'predictions': [],
        'start_time': None
    }

# === 開始新場 ===
if st.button("開始監測本場", type="primary"):
    st.session_state.ucb_state = {
        't': 0,
        'selected_count': {},
        'momentum_history': {},
        'odds_history': {},
        'investment_history': {},
        'predictions': [],
        'start_time': datetime.now()
    }
    st.rerun()

# === 主循環：每 30 秒更新 ===
placeholder = st.empty()
if st.session_state.ucb_state['start_time']:
    with placeholder.container():
        state = st.session_state.ucb_state
        now = datetime.now()
        elapsed = (now - state['start_time']).total_seconds()

        # 每 30 秒更新
        if state['t'] == 0 or elapsed - state['t']*30 >= 30:
            state['t'] += 1
            t = state['t']

            # 抓資料
            odds_data = get_odds_data(Date, place, race_no, methodlist)
            inv_data = get_investment_data(Date, place, race_no, methodlist)

            if 'WIN' not in odds_data or not odds_data['WIN']:
                st.warning("無 WIN 賠率資料")
                st.stop()

            win_odds = np.array([o if np.isfinite(o) else 999 for o in odds_data['WIN']])
            total_inv = inv_data.get('WIN', [0])[0] if inv_data.get('WIN') else 0

            # 計算每匹馬的隱含投注量
            inv_per_horse = total_inv / win_odds if total_inv > 0 else np.zeros_like(win_odds)
            inv_per_horse = np.nan_to_num(inv_per_horse, nan=0)

            # 初始化
            n_horses = len(win_odds)
            horses = list(range(1, n_horses + 1))
            if not state['selected_count']:
                state['selected_count'] = {h: 0 for h in horses}
                state['momentum_history'] = {h: [] for h in horses}

            # 計算動量（變化率）
            momentum = {}
            for h in horses:
                idx = h - 1
                state['momentum_history'][h].append(inv_per_horse[idx])
                hist = state['momentum_history'][h]
                if len(hist) >= 2:
                    momentum[h] = hist[-1] - hist[-2]  # 動量 = 增量
                else:
                    momentum[h] = 0

            # UCB 計算
            ucb_values = {}
            for h in horses:
                n = max(state['selected_count'][h], 1)
                exploration = c_weight * np.sqrt(np.log(t) / n)
                ucb = momentum[h] + exploration
                ucb_values[h] = ucb

            # 選擇
            selected = max(ucb_values, key=ucb_values.get)
            state['selected_count'][selected] += 1
            state['predictions'].append({
                't': t,
                'selected': selected,
                'odds': win_odds[selected-1],
                'momentum': momentum[selected],
                'ucb': ucb_values[selected]
            })

            # === 即時顯示 ===
            st.subheader(f"第 {race_no} 場 - 更新 {t} ({t*30}秒)")
            colA, colB = st.columns(2)

            with colA:
                df = pd.DataFrame([
                    {
                        '馬號': h,
                        '賠率': f"{win_odds[h-1]:.2f}",
                        '動量': f"{momentum[h]:+.1f}",
                        '被選': state['selected_count'][h],
                        'UCB': f"{ucb_values[h]:.3f}"
                    }
                    for h in horses
                ])
                st.dataframe(df.style.highlight_max(subset=['UCB'], color='lightgreen'))

            with colB:
                if len(state['predictions']) > 1:
                    pred_df = pd.DataFrame(state['predictions'])
                    fig = go.Figure()
                    for h in horses:
                        counts = [1 if p['selected']==h else 0 for p in state['predictions']]
                        cum = np.cumsum(counts)
                        fig.add_trace(go.Scatter(x=list(range(1,len(cum)+1)), y=cum, name=f"馬 {h}"))
                    fig.update_layout(title="UCB 預測趨勢", xaxis_title="更新次數", yaxis_title="被選累計")
                    st.plotly_chart(fig)

            # 最終預測（賽前 1 分鐘）
            if t >= 10:  # 5 分鐘後鎖定
                final_horse = max(state['selected_count'], key=state['selected_count'].get)
                st.success(f"最終預測：**馬 {final_horse}** (被選 {state['selected_count'][final_horse]} 次)")
                st.info(f"建議投注：WIN 馬 {final_horse} @ {win_odds[final_horse-1]:.2f}")

            st.rerun()
