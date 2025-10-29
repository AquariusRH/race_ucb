import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta as datere
import pytz
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.font_manager.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
matplotlib.rc('font', family='Taipei Sans TC Beta')
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# Show the page title and description.
st.set_page_config(page_title="Jockey Race")
st.title("Jockey Race 賽馬程式")

# @title 2. {func} 下載數據
# @title 處理數據
def get_investment_data():
  url = 'https://info.cld.hkjc.com/graphql/base/'
  headers = {'Content-Type': 'application/json'}

  payload_investment = {
      "operationName": "racing",
      "variables": {
          "date": str(Date),
          "venueCode": place,
          "raceNo": int(race_no),
          "oddsTypes": methodlist
      },
      "query": """
      query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
        raceMeetings(date: $date, venueCode: $venueCode) {
          totalInvestment
          poolInvs: pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
            id
            leg {
              number
              races
            }
            status
            sellStatus
            oddsType
            investment
            mergedPoolId
            lastUpdateTime
          }
        }
      }
      """
  }

  response = requests.post(url, headers=headers, json=payload_investment)

  if response.status_code == 200:
      investment_data = response.json()

      # Extracting the investment into different types of oddsType
      investments = {
          "WIN": [],
          "PLA": [],
          "QIN": [],
          "QPL": [],
          "FCT": [],
          "TRI": [],
          "FF": []
      }

      race_meetings = investment_data.get('data', {}).get('raceMeetings', [])
      if race_meetings:
          for meeting in race_meetings:
              pool_invs = meeting.get('poolInvs', [])
              for pool in pool_invs:
                  if place not in ['ST','HV']:
                    id = pool.get('id')
                    if id[8:10] != place:
                      continue                
                  investment = float(pool.get('investment'))
                  investments[pool.get('oddsType')].append(investment)

          #print("Investments:", investments)
      else:
          print("No race meetings found in the response.")

      return investments
  else:
      print(f"Error: {response.status_code}")

def get_odds_data():
  url = 'https://info.cld.hkjc.com/graphql/base/'
  headers = {'Content-Type': 'application/json'}
  payload_odds = {
      "operationName": "racing",
      "variables": {
          "date": str(Date),
          "venueCode": place,
          "raceNo": int(race_no),
          "oddsTypes": methodlist
      },
      "query": """
      query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
        raceMeetings(date: $date, venueCode: $venueCode) {
          pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
            id
            status
            sellStatus
            oddsType
            lastUpdateTime
            guarantee
            minTicketCost
            name_en
            name_ch
            leg {
              number
              races
            }
            cWinSelections {
              composite
              name_ch
              name_en
              starters
            }
            oddsNodes {
              combString
              oddsValue
              hotFavourite
              oddsDropValue
              bankerOdds {
                combString
                oddsValue
              }
            }
          }
        }
      }
      """
  }

  response = requests.post(url, headers=headers, json=payload_odds)
  if response.status_code == 200:
      odds_data = response.json()
          # Extracting the oddsValue into different types of oddsType and sorting by combString for QIN and QPL
      # Initialize odds_values with empty lists for each odds type
      odds_values = {
          "WIN": [],
          "PLA": [],
          "QIN": [],
          "QPL": [],
          "FCT": [],
          "TRI": [],
          "FF": []
      }
      
      race_meetings = odds_data.get('data', {}).get('raceMeetings', [])
      for meeting in race_meetings:
          pm_pools = meeting.get('pmPools', [])
          for pool in pm_pools:
              if place not in ['ST', 'HV']:
                  id = pool.get('id')
                  if id and id[8:10] != place:  # Check if id exists before slicing
                      continue
              odds_nodes = pool.get('oddsNodes', [])
              odds_type = pool.get('oddsType')
              odds_values[odds_type] = []
              # Skip if odds_type is invalid or not in odds_values
              if not odds_type or odds_type not in odds_values:
                  continue
              for node in odds_nodes:
                  oddsValue = node.get('oddsValue')
                  # Skip iteration if oddsValue is None, empty, or '---'
                  if oddsValue == 'SCR':
                      oddsValue = np.inf
                  else:
                      try:
                          oddsValue = float(oddsValue)
                      except (ValueError, TypeError):
                          continue  # Skip if oddsValue can't be converted to float
                  # Store data based on odds_type
                  if odds_type in ["QIN", "QPL", "FCT", "TRI", "FF"]:
                      comb_string = node.get('combString')
                      if comb_string:  # Ensure combString exists
                          odds_values[odds_type].append((comb_string, oddsValue))
                  else:
                      odds_values[odds_type].append(oddsValue)
      # Sorting the odds values for specific types by combString in ascending order
      for odds_type in ["QIN", "QPL", "FCT", "TRI", "FF"]:
          odds_values[odds_type].sort(key=lambda x: x[0], reverse=False)
      return odds_values

      #print("WIN Odds Values:", odds_values["WIN"])
      #print("PLA Odds Values:", odds_values["PLA"])
      #print("QIN Odds Values (sorted by combString):", [value for _, value in odds_values["QIN"]])
      #print("QPL Odds Values (sorted by combString):", [value for _, value in odds_values["QPL"]])

  else:
      print(f"Error: {response.status_code}")

def save_odds_data(time_now,odds):
  for method in methodlist:
      if method in ['WIN', 'PLA']:
        if odds_dict[method].empty:
            # Initialize the DataFrame with the correct number of columns
            odds_dict[method] = pd.DataFrame(columns=np.arange(1, len(odds[method]) + 1))
        odds_dict[method].loc[time_now] = odds[method]
      elif method in ['QIN','QPL',"FCT","TRI","FF"]:
        if odds[method]:
          combination, odds_array = zip(*odds[method])
          if odds_dict[method].empty:
            odds_dict[method] = pd.DataFrame(columns=combination)
            # Set the values with the specified index
          odds_dict[method].loc[time_now] = odds_array
  #st.write(odds_dict)

def save_investment_data(time_now,investment,odds):
  for method in methodlist:
      if method in ['WIN', 'PLA']:
        if investment_dict[method].empty:
            # Initialize the DataFrame with the correct number of columns
            investment_dict[method] = pd.DataFrame(columns=np.arange(1, len(odds[method]) + 1))
        investment_df = [round(investments[method][0]  / 1000 / odd, 2) for odd in odds[method]]
        investment_dict[method].loc[time_now] = investment_df
      elif method in ['QIN','QPL',"FCT","TRI","FF"]:
        if odds[method]:
          combination, odds_array = zip(*odds[method])
          if investment_dict[method].empty:
            investment_dict[method] = pd.DataFrame(columns=combination)
          investment_df = [round(investments[method][0]  / 1000 / odd, 2) for odd in odds_array]
              # Set the values with the specified index
          investment_dict[method].loc[time_now] = investment_df
  #st.write(investment_dict)
def print_data(time_now,period):
  for watch in watchlist:
    data = odds_dict[watch].tail(period)
    data.index = data.index.strftime('%H:%M:%S')
    if watch in ['WIN','PLA']:
      data.columns = np.arange(len(numbered_dict[race_no]))+1
      pd.set_option('display.max_colwidth', 10)
    with pd.option_context('display.max_rows', None, 'display.max_columns',None):  # more options can be specified also
        name = methodCHlist[methodlist.index(watch)]
        print(f'{name} 賠率')
        data



def investment_combined(time_now,method,df):
  sums = {}
  for col in df.columns:
      # Split the column name to get the numbers
      num1, num2 = col.split(',')
      # Convert to integers
      num1, num2 = int(num1), int(num2)

      # Sum the column values
      col_sum = df[col].sum()

      # Add the sum to the corresponding numbers in the dictionary
      if num1 in sums:
          sums[num1] += col_sum
      else:
          sums[num1] = col_sum

      if num2 in sums:
          sums[num2] += col_sum
      else:
          sums[num2] = col_sum

  # Convert the sums dictionary to a dataframe for better visualization
  sums_df = pd.DataFrame([sums],index = [time_now]) /2
  return sums_df

def get_overall_investment(time_now,dict):
    investment_df = investment_dict
    no_of_horse = len(investment_df['WIN'].columns)
    total_investment_df = pd.DataFrame(index =[time_now], columns=np.arange(1,no_of_horse +1))
    for method in methodlist:
      if method in ['WIN','PLA']:
        overall_investment_dict[method] = overall_investment_dict[method]._append(investment_dict[method].tail(1))
      elif method in ['QIN','QPL']:
        if not investment_df[method].empty:
          overall_investment_dict[method] = overall_investment_dict[method]._append(investment_combined(time_now,method,investment_dict[method].tail(1)))
        else:
          continue

    for horse in range(1,no_of_horse+1):
        total_investment = 0
        for method in methodlist:
            if method in ['WIN', 'PLA']:
                investment = overall_investment_dict[method][horse].values[-1]
            elif method in ['QIN','QPL']:
              if not investment_df[method].empty: 
                investment = overall_investment_dict[method][horse].values[-1]
              else:
                continue
            total_investment += investment
        total_investment_df[horse] = total_investment
    overall_investment_dict['overall'] = overall_investment_dict['overall']._append(total_investment_df)

def print_bar_chart(time_now):
  post_time = post_time_dict[race_no]
  time_25_minutes_before = np.datetime64(post_time - timedelta(minutes=25) + timedelta(hours=8))
  time_5_minutes_before = np.datetime64(post_time - timedelta(minutes=5) + timedelta(hours=8))
  
  for method in print_list:
      odds_list = pd.DataFrame()
      df = pd.DataFrame()
      if method == 'overall':
          df = overall_investment_dict[method]
          change_data = diff_dict[method].iloc[-1]
      elif method in methodlist:
          df = overall_investment_dict[method]
          change_data = diff_dict[method].tail(10).sum(axis = 0)
          odds_list = odds_dict[method]
      if df.empty:
        continue
      fig, ax1 = plt.subplots(figsize=(12, 6))
      df.index = pd.to_datetime(df.index)
      df_1st = pd.DataFrame()
      df_1st_2nd = pd.DataFrame()
      df_2nd = pd.DataFrame()
      #df_3rd = pd.DataFrame()
      df_1st = df[df.index< time_25_minutes_before].tail(1)
      df_1st_2nd = df[df.index >= time_25_minutes_before].head(1)
      df_2nd = df[df.index >= time_25_minutes_before].tail(1)
      df_3rd = df[df.index>= time_5_minutes_before].tail(1)

      change_df = pd.DataFrame([change_data.apply(lambda x: x*6 if x > 0 else x*3)],columns=change_data.index,index =[df.index[-1]])
      print(change_df)
      if method in ['WIN', 'PLA']:
        odds_list.index = pd.to_datetime(odds_list.index)
        odds_1st = odds_list[odds_list.index< time_25_minutes_before].tail(1)
        odds_2nd = odds_list[odds_list.index >= time_25_minutes_before].tail(1)
        #odds_3rd = odds_list[odds_list.index>= time_5_minutes_before].tail(1)

      bars_1st = None
      bars_2nd = None
      #bars_3rd = None
      # Initialize data_df
      if not df_1st.empty:
          data_df = df_1st
          data_df = data_df._append(df_2nd)
      elif not df_1st_2nd.empty:
          data_df = df_1st_2nd
          if not df_2nd.empty and not df_2nd.equals(df_1st_2nd):  # Avoid appending identical df_2nd
              data_df = data_df._append(df_2nd)
      else:
          data_df = pd.DataFrame()  # Fallback if both are empty
      #final_data_df = data_df._append(df_3rd)
      final_data_df = data_df
      sorted_final_data_df = final_data_df.sort_values(by=final_data_df.index[0], axis=1, ascending=False)
      diff = sorted_final_data_df.diff().dropna()
      diff[diff < 0] = 0
      X = sorted_final_data_df.columns
      X_axis = np.arange(len(X))
      sorted_change_df = change_df[X]
      if df_3rd.empty:
                  bar_colour = 'blue'
      else:
                  bar_colour = 'red'
      if not df_1st.empty:
          if df_2nd.empty:
                bars_1st = ax1.bar(X_axis, sorted_final_data_df.iloc[0], 0.4, label='投注額', color='pink')
          else:
                bars_2nd = ax1.bar(X_axis - 0.2, sorted_final_data_df.iloc[1], 0.4, label='25分鐘', color=bar_colour)
                bar = ax1.bar(X_axis+0.2,sorted_change_df.iloc[0],0.4,label='改變',color='grey')
                #if not df_3rd.empty:
                    #bars_3rd = ax1.bar(X_axis, diff.iloc[0], 0.3, label='5分鐘', color='red')
      else:
            if df_2nd.equals(df_1st_2nd):
              bars_2nd = ax1.bar(X_axis - 0.2, sorted_final_data_df.iloc[0], 0.4, label='25分鐘', color=bar_colour)
            else:
                bars_2nd = ax1.bar(X_axis - 0.2, sorted_final_data_df.iloc[1], 0.4, label='25分鐘', color=bar_colour)
                bar = ax1.bar(X_axis+0.2,sorted_change_df.iloc[0],0.4,label='改變',color='grey')
                #if not df_3rd.empty:
                    #bars_3rd = ax1.bar(X_axis, diff.iloc[0], 0.3, label='5分鐘', color='red')
            #else:
                #bars_3rd = ax1.bar(X_axis-0.2, sorted_final_data_df.iloc[0], 0.4, label='5分鐘', color='red')
                #bar = ax1.bar(X_axis+0.2,sorted_change_df.iloc[0],0.4,label='改變',color='grey')

      # Add numbers above bars
      if method in ['WIN', 'PLA']:
        if bars_2nd is not None:
          sorted_odds_list_2nd = odds_2nd[X].iloc[0]
          for bar, odds in zip(bars_2nd, sorted_odds_list_2nd):
              yval = bar.get_height()
              ax1.text(bar.get_x() + bar.get_width() / 2, yval, odds, ha='center', va='bottom')
        #if bars_3rd is not None:
          #sorted_odds_list_3rd = odds_3rd[X].iloc[0]
          #for bar, odds in zip(bars_3rd, sorted_odds_list_3rd):
               # yval = bar.get_height()
                #ax1.text(bar.get_x() + bar.get_width() / 2, yval, odds, ha='center', va='bottom')
        elif bars_1st is not None:
          sorted_odds_list_1st = odds_1st[X].iloc[0]
          for bar, odds in zip(bars_1st, sorted_odds_list_1st):
              yval = bar.get_height()
              ax1.text(bar.get_x() + bar.get_width() / 2, yval, odds, ha='center', va='bottom')

      namelist_sort = [numbered_dict[race_no][i - 1] for i in X]
      formatted_namelist = [label.split('.')[0] + '.' + '\n'.join(label.split('.')[1]) for label in namelist_sort]
      plt.xticks(X_axis, formatted_namelist, fontsize=12)
      ax1.grid(color='lightgrey', axis='y', linestyle='--')
      ax1.set_ylabel('投注額',fontsize=15)
      ax1.tick_params(axis='y')
      fig.legend()

      if method == 'overall':
          plt.title('綜合', fontsize=15)
      elif method == 'QIN':
          plt.title('連贏', fontsize=15)
      elif method == 'QPL':
          plt.title('位置Q', fontsize=15)
      elif method == 'WIN':
          plt.title('獨贏', fontsize=15)
      elif method == 'PLA':
          plt.title('位置', fontsize=15)
      st.pyplot(fig)

def weird_data(investments):

  for method in methodlist:
    if investment_dict[method].empty:
      continue
    latest_investment = investment_dict[method].tail(1).values
    last_time_odds = odds_dict[method].tail(2).head(1)
    expected_investment = investments[method][0] / 1000 / last_time_odds
    diff = round(latest_investment - expected_investment,0)
    if method in ['WIN','PLA']:
        diff_dict[method] = diff_dict[method]._append(diff)
    elif method in ['QIN','QPL']:
        diff_dict[method] = diff_dict[method]._append(investment_combined(time_now,method,diff))
    #benchmark = benchmark_dict.get(method)
    #diff.index = diff.index.strftime('%H:%M:%S')
    #for index in investment_dict[method].tail(1).columns:
      #error = diff[index].values[0]
      #error_df = []
      #if error > benchmark:
        #if error < benchmark * 2 :
         # highlight = '-'
        #elif error < benchmark * 3 :
        #  highlight = '*'
       # elif error < benchmark * 4 :
        #  highlight = '**'
        #else:
        #  highlight = '***'
        #error_df = pd.DataFrame([[index,error,odds_dict[method].tail(1)[index].values,highlight]], columns=['No.', 'error','odds', 'Highlight'],index = diff.index)
      #weird_dict[method] = weird_dict[method]._append(error_df)

def change_overall(time_now):
  total_investment = 0
  for method in methodlist:
    total_investment += diff_dict[method].sum(axis=0)
  total_investment_df = pd.DataFrame([total_investment],index = [time_now])
  diff_dict['overall'] = diff_dict['overall']._append(total_investment_df)

def print_concern_weird_dict():
    target_list = methodlist[0:4]
    if 'QPL' not in target_list:
      target_list = methodlist[0:3]
    for method in target_list:
      name = methodCHlist[methodlist.index(method)]
      st.write(f'{name} 異常投注')
      df = weird_dict[method]
      df_tail = df.tail(20)[::-1]
      count = df.value_counts('No.')
      count_df = count.to_frame().T

      # Create two columns
      col1, col2 = st.columns(2)

      # Display df_tail in the first column
      with col1:
         df_tail

      # Display count_df in the second column
      with col2:
          count_df
# Define a function to apply conditional formatting
def highlight_change(val):
    color = 'limegreen' if '+' in val else 'crimson' if '-' in val else ''
    return f'color: {color}'

def top(method_odds_df, method_investment_df, method):
    # Extract the first row from odds DataFrame
    first_row_odds = method_odds_df.iloc[0]
    first_row_odds_df = first_row_odds.to_frame(name='Odds').reset_index()
    first_row_odds_df.columns = ['Combination', 'Odds']

    # Extract the last row from odds DataFrame
    last_row_odds = method_odds_df.iloc[-1]
    last_row_odds_df = last_row_odds.to_frame(name='Odds').reset_index()
    last_row_odds_df.columns = ['Combination', 'Odds']
    third_last_row_index = max(-len(method_odds_df), -11)
    third_last_row_odds = method_odds_df.iloc[third_last_row_index]
    third_last_row_odds_df = third_last_row_odds.to_frame(name='Odds').reset_index()
    third_last_row_odds_df.columns = ['Combination', 'Odds']
    # Extract the second last row from odds DataFrame (or the closest available row)
    second_last_row_index = max(-len(method_odds_df), -3)
    second_last_row_odds = method_odds_df.iloc[second_last_row_index]
    second_last_row_odds_df = second_last_row_odds.to_frame(name='Odds').reset_index()
    second_last_row_odds_df.columns = ['Combination', 'Odds']

    # Calculate the initial rank and initial odds
    first_row_odds_df['Initial_Rank'] = first_row_odds_df['Odds'].rank(method='min').astype(int)
    first_row_odds_df['Initial_Odds'] = first_row_odds_df['Odds']

    # Calculate the current rank and current odds
    last_row_odds_df['Current_Rank'] = last_row_odds_df['Odds'].rank(method='min').astype(int)
    last_row_odds_df['Initial_Rank'] = first_row_odds_df['Initial_Rank'].values
    last_row_odds_df['Initial_Odds'] = first_row_odds_df['Initial_Odds'].values

    # Calculate the previous rank using the second last row
    second_last_row_odds_df['Previous_Rank'] = second_last_row_odds_df['Odds'].rank(method='min').astype(int)
    last_row_odds_df['Previous_Rank'] = second_last_row_odds_df['Previous_Rank'].values

    # Calculate the change of rank
    last_row_odds_df['Change_of_Rank'] = last_row_odds_df['Initial_Rank'] - last_row_odds_df['Current_Rank']
    last_row_odds_df['Change_of_Rank'] = last_row_odds_df['Change_of_Rank'].apply(lambda x: f'+{x}' if x > 0 else (str(x) if x < 0 else '0'))

    # Combine the initial rank and change of rank into the same column format like 10 (+1)
    last_row_odds_df['Initial_Rank'] = last_row_odds_df.apply(lambda row: f"{row['Initial_Rank']}" f"({row['Change_of_Rank']})", axis=1)

    # Calculate the difference between the current rank and previous rank and add this difference to the previous rank in the format 10 (+1)
    last_row_odds_df['Change_of_Previous_Rank'] = last_row_odds_df['Previous_Rank'] - last_row_odds_df['Current_Rank']
    last_row_odds_df['Change_of_Previous_Rank'] = last_row_odds_df['Change_of_Previous_Rank'].apply(lambda x: f'+{x}' if x > 0 else (str(x) if x < 0 else '0'))
    last_row_odds_df['Previous_Rank'] = last_row_odds_df.apply(lambda row: f"{row['Previous_Rank']}" f"({row['Change_of_Previous_Rank']})", axis=1)

    # Rearrange the columns as requested
    final_df = last_row_odds_df[['Combination', 'Odds', 'Initial_Odds', 'Current_Rank', 'Initial_Rank', 'Previous_Rank']]

    # Format the odds to one decimal place using .loc to avoid SettingWithCopyWarning
    final_df.loc[:, 'Odds'] = final_df['Odds'].round(1)
    final_df.loc[:, 'Initial_Odds'] = final_df['Initial_Odds'].round(1)

    # Extract the first row from investment DataFrame
    first_row_investment = method_investment_df.iloc[0]
    first_row_investment_df = first_row_investment.to_frame(name='Investment').reset_index()
    first_row_investment_df.columns = ['Combination', 'Investment']

    # Extract the last row from investment DataFrame
    last_row_investment = method_investment_df.iloc[-1]
    last_row_investment_df = last_row_investment.to_frame(name='Investment').reset_index()
    last_row_investment_df.columns = ['Combination', 'Investment']

    # Extract the second last row from investment DataFrame (or the closest available row)
    second_last_row_index = max(-len(method_investment_df), -3)
    second_last_row_investment = method_investment_df.iloc[second_last_row_index]
    second_last_row_investment_df = second_last_row_investment.to_frame(name='Investment').reset_index()
    second_last_row_investment_df.columns = ['Combination', 'Investment']
    third_last_row_index = max(-len(method_investment_df), -7)
    third_last_row_investment = method_investment_df.iloc[third_last_row_index]
    third_last_row_investment_df = third_last_row_investment.to_frame(name='Investment').reset_index()
    third_last_row_investment_df.columns = ['Combination', 'Investment']
    # Calculate the difference in investment before sorting
    last_row_investment_df['Investment_Change'] = last_row_investment_df['Investment'] - first_row_investment_df['Investment'].values
    last_row_investment_df['Investment_Change'] = last_row_investment_df['Investment_Change'].apply(lambda x: x if x > 0 else 0)
    second_last_row_investment_df['Previous_Investment_Change'] = last_row_investment_df['Investment'] - second_last_row_investment_df['Investment'].values
    second_last_row_investment_df['Previous_Investment_Change'] = second_last_row_investment_df['Previous_Investment_Change'].apply(lambda x: x if x > 0 else 0)
    third_last_row_investment_df['Previous_Investment_Change'] = last_row_investment_df['Investment'] - third_last_row_investment_df['Investment'].values
    third_last_row_investment_df['Previous_Investment_Change'] = third_last_row_investment_df['Previous_Investment_Change'].apply(lambda x: x if x > 0 else 0)

    # Sort the final DataFrame by odds value
    final_df = final_df.sort_values(by='Odds')

    # Combine the investment data with the final DataFrame based on the combination
    final_df = final_df.merge(last_row_investment_df[['Combination', 'Investment_Change', 'Investment']], on='Combination', how='left')
    final_df = final_df.merge(second_last_row_investment_df[['Combination', 'Previous_Investment_Change']], on='Combination', how='left')
    final_df = final_df.merge(third_last_row_investment_df[['Combination', 'Previous_Investment_Change']], on='Combination', how='left')

    if method in ['WIN','PLA']:
      final_df.columns = ['馬匹', '賠率', '最初賠率', '排名', '最初排名', '上一次排名', '投注變化', '投注', '一分鐘投注','三分鐘投注']
      target_df = final_df
      rows_with_plus = target_df[
          target_df['最初排名'].astype(str).str.contains('\+') |
          target_df['上一次排名'].astype(str).str.contains('\+')
      ][['馬匹', '賠率', '最初排名', '上一次排名']]
      # Apply the conditional formatting to the 初始排名 and 前一排名 columns and add a bar to the 投資變化 column
      styled_df = final_df.style.format({
        '賠率': '{:.1f}',
        '最初賠率': '{:.1f}',
        '投注變化': '{:.2f}k',
        '投注': '{:.2f}k',
        '一分鐘投注': '{:.2f}k',
        '三分鐘投注': '{:.2f}k'
      }).map(highlight_change, subset=['最初排名', '上一次排名']).bar(subset=['投注變化', '一分鐘投注','三分鐘投注'], color='rgba(173, 216, 230, 0.5)').hide(axis='index')
      styled_rows_with_plus = rows_with_plus.style.format({'賠率': '{:.1f}'}).map(highlight_change, subset=['最初排名', '上一次排名']).hide(axis='index')
      # Display the styled DataFrame
      st.write(styled_df.to_html(), unsafe_allow_html=True)
      st.write(styled_rows_with_plus.to_html(), unsafe_allow_html=True)


    else:
      final_df.columns = ['組合', '賠率', '最初賠率', '排名', '最初排名', '上一次排名', '投注變化', '投注', '一分鐘投注','三分鐘投注']
      target_df = final_df.head(15)
      target_special_df = final_df.head(30)
      rows_with_plus = target_special_df[
          target_special_df['最初排名'].astype(str).str.contains('\+') |
          target_special_df['上一次排名'].astype(str).str.contains('\+')
      ][['組合', '賠率', '最初排名', '上一次排名']]
    

      # Apply the conditional formatting to the 初始排名 and 前一排名 columns and add a bar to the 投資變化 column
      styled_df = target_df.style.format({
        '賠率': '{:.1f}',
        '最初賠率': '{:.1f}',
        '投注變化': '{:.2f}k',
        '投注': '{:.2f}k',
        '一分鐘投注': '{:.2f}k',
        '三分鐘投注': '{:.2f}k'
      }).map(highlight_change, subset=['最初排名', '上一次排名']).bar(subset=['投注變化', '一分鐘投注','三分鐘投注'], color='rgba(173, 216, 230, 0.5)').hide(axis='index')
      styled_rows_with_plus = rows_with_plus.style.format({'賠率': '{:.1f}'}).map(highlight_change, subset=['最初排名', '上一次排名']).hide(axis='index')
      # Display the styled DataFrame
      st.write(styled_df.to_html(), unsafe_allow_html=True)

      if method in ["QIN","QPL","FCT","TRI","FF"]:
        if method in ["QIN"]:
          notice_df = final_df[(final_df['一分鐘投注'] >= 100) | (final_df['三分鐘投注'] >= 300)][['組合', '賠率', '一分鐘投注', '三分鐘投注']]
        elif method in ["QPL"]:
          notice_df = final_df[(final_df['一分鐘投注'] >= 200) | (final_df['三分鐘投注'] >= 600)][['組合', '賠率', '一分鐘投注', '三分鐘投注']]
        elif method in ["FCT"]:
          notice_df = final_df[(final_df['一分鐘投注'] >= 10) | (final_df['三分鐘投注'] >= 30)][['組合', '賠率', '一分鐘投注', '三分鐘投注']]
        else:
          notice_df = final_df[(final_df['一分鐘投注'] >= 5) | (final_df['三分鐘投注'] >= 15)][['組合', '賠率', '一分鐘投注', '三分鐘投注']]
        styled_notice_df = notice_df.style.format({'賠率': '{:.1f}','一分鐘投注': '{:.2f}k','三分鐘投注': '{:.2f}k'}).bar(subset=['一分鐘投注','三分鐘投注'], color='rgba(173, 216, 230, 0.5)').hide(axis='index')
        

      col1, col2 = st.columns(2)
      with col1:
        st.write(styled_rows_with_plus.to_html(), unsafe_allow_html=True)
      with col2:
        st.write(styled_notice_df.to_html(), unsafe_allow_html=True)

def print_top():
  for method in top_list:
        if odds[method]:
          methodCHlist[methodlist.index(method)]
          top(odds_dict[method], investment_dict[method], method)

def print_highlight():
  for method in ['WIN','QIN']:
    df = weird_dict[method]
    if not df.empty:
      filtered_df_3 = df[df['Highlight'] == '***']
      filtered_df_2 = df[df['Highlight'] == '**']
      filtered_df_1 = df[df['Highlight'] == '*']
      if method == 'WIN':
        st.write('獨贏')
      elif method == 'QIN':
        st.write('連贏')
      highlightColumns = st.columns(3)
      if not filtered_df_3.empty:
        with highlightColumns[0]:
          crosstab_3 = pd.crosstab(filtered_df_3['No.'],filtered_df_3['Highlight']).sort_values(by='***', ascending=False)
          crosstab_3
      if not filtered_df_2.empty:  
        with highlightColumns[1]:
          crosstab_2 = pd.crosstab(filtered_df_2['No.'],filtered_df_2['Highlight']).sort_values(by='**', ascending=False)
          crosstab_2
      if not filtered_df_1.empty:  
        with highlightColumns[2]:
          crosstab_1 = pd.crosstab(filtered_df_1['No.'],filtered_df_1['Highlight']).sort_values(by='*', ascending=False)
          crosstab_1

def run_ucb_prediction(win_odds, total_win_inv, horses, state, t):
    n_horses = len(win_odds)
    inv_per_horse = total_win_inv / win_odds if total_win_inv > 0 else np.zeros(n_horses)
    inv_mean = np.mean(inv_per_horse) if np.mean(inv_per_horse) > 0 else 1

    # 正規化動量
    momentum = {}
    for h in horses:
        idx = h - 1
        state['momentum_hist'][h].append(inv_per_horse[idx])
        hist = state['momentum_hist'][h]
        raw = hist[-1] - hist[-2] if len(hist) >= 2 else 0
        momentum[h] = raw / (inv_mean + 1e-6)

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
    top4 = sorted(ucb_values, key=ucb_values.get, reverse=True)[:4]
    for h in top4:
        state['selected_count'][h] += 1

    # 建表格
    table_data = []
    for i, h in enumerate(horses):
        table_data.append({
            '馬號': h,
            '馬名': race_dict[race_no]['馬名'][i],
            '騎師': race_dict[race_no]['騎師'][i],
            '賠率': f"{win_odds[i]:.2f}",
            '動量': f"{momentum.get(h, 0):+.3f}",
            '被選次數': state['selected_count'][h],
            'UCB': f"{ucb_values.get(h, 0):.3f}",
            '排名': f"Top {top4.index(h)+1}" if h in top4 else ""
        })
    df = pd.DataFrame(table_data).sort_values('UCB', ascending=False)
    return df, top4

def main(time_now,odds,investments,period):
  save_odds_data(time_now,odds)
  save_investment_data(time_now,investments,odds)
  get_overall_investment(time_now,investments)
  weird_data(investments)
  change_overall(time_now)
  print_bar_chart(time_now)
  print_top()

# Display the date picker widget
infoColumns = st.columns(3)
with infoColumns[0]:
    Date = st.date_input('日期:', value=datetime.now())
with infoColumns[1]:
    options = ['ST', 'HV', 'S1', 'S2', 'S3', 'S4', 'S5']
    place = st.selectbox('場地:', options)
with infoColumns[2]:
    race_options = np.arange(1, 12)
    race_no = st.selectbox('場次:', race_options)

# Initialize lists (using list2 and list2_ch as default; change to list1 and list1_ch if preferred)
available_methods = ['WIN', 'PLA', 'QIN', 'QPL', 'FCT', 'TRI', 'FF']
available_methods_ch = ['獨贏', '位置', '連贏', '位置Q', '二重彩', '單T', '四連環']
print_list_default = ['WIN','PLA','QIN','QPL','FCT', 'TRI', 'FF']
top_list_default = ['QIN','QPL','WIN','PLA','FCT', 'TRI', 'FF']
default_checked_methods = ['WIN','QIN']
# Initialize session state variables
if 'reset' not in st.session_state:
    st.session_state.reset = False
if 'api_called' not in st.session_state:
    st.session_state.api_called = False

# Create individual checkboxes for each betting method
st.write("選擇投注方式 (Select Betting Methods):")
num_methods = len(available_methods)
method_columns = st.columns(num_methods)  # Create one column per method
selected_methods = []
for idx, (method, method_ch) in enumerate(zip(available_methods, available_methods_ch)):
  with method_columns[idx]:
    is_default_checked = method in default_checked_methods
    if st.checkbox(method_ch, value=is_default_checked, key=method):
        selected_methods.append(method)
# Update methodlist and methodCHlist based on selections
methodlist = selected_methods

methodCHlist = [available_methods_ch[available_methods.index(method)] for method in selected_methods]

# Update print_list based on selections (only include selected methods that are in the default print_list)
print_list = [item for item in print_list_default if item in selected_methods ]
top_list = [item for item in top_list_default if item in selected_methods ]
# Save changes to race_no (assuming race_no is defined elsewhere)
race_no_value = race_no if 'race_no' in globals() else None

# Example benchmark dictionary (assuming these variables are defined elsewhere)
benchmark_dict = {
    "WIN": benchmark_win if 'benchmark_win' in globals() else None,
    "PLA": benchmark_pla if 'benchmark_pla' in globals() else None,
    "QIN": benchmark_qin if 'benchmark_qin' in globals() else None,
    "QPL": benchmark_qpl if 'benchmark_qpl' in globals() else None
}

# Define the button callback
def click_start_button():
    st.session_state.reset = True

# Add a button to trigger an action
if st.button("開始", on_click=click_start_button):
    st.write(f"Selected methods: {methodlist}")
    st.write(f"Chinese labels: {methodCHlist}")
    st.write(f"Print list: {print_list}")

# Display current state for debugging
#st.write("Current methodlist:", methodlist)
#st.write("Current methodCHlist:", methodCHlist)
#st.write("Current print_list:", print_list)

if not st.session_state.api_called:
  url = 'https://info.cld.hkjc.com/graphql/base/'
  headers = {'Content-Type': 'application/json'}
  payload = {
      "operationName": "raceMeetings",
      "variables": {"date": str(Date), "venueCode": place},
      "query": """
      fragment raceFragment on Race {
        id
        no
        status
        raceName_en
        raceName_ch
        postTime
        country_en
        country_ch
        distance
        wageringFieldSize
        go_en
        go_ch
        ratingType
        raceTrack {
          description_en
          description_ch
        }
        raceCourse {
          description_en
          description_ch
          displayCode
        }
        claCode
        raceClass_en
        raceClass_ch
        judgeSigns {
          value_en
        }
      }
  
      fragment racingBlockFragment on RaceMeeting {
        jpEsts: pmPools(
          oddsTypes: [TCE, TRI, FF, QTT, DT, TT, SixUP]
          filters: ["jackpot", "estimatedDividend"]
        ) {
          leg {
            number
            races
          }
          oddsType
          jackpot
          estimatedDividend
          mergedPoolId
        }
        poolInvs: pmPools(
          oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
        ) {
          id
          leg {
            races
          }
        }
        penetrometerReadings(filters: ["first"]) {
          reading
          readingTime
        }
        hammerReadings(filters: ["first"]) {
          reading
          readingTime
        }
        changeHistories(filters: ["top3"]) {
          type
          time
          raceNo
          runnerNo
          horseName_ch
          horseName_en
          jockeyName_ch
          jockeyName_en
          scratchHorseName_ch
          scratchHorseName_en
          handicapWeight
          scrResvIndicator
        }
      }
  
      query raceMeetings($date: String, $venueCode: String) {
        timeOffset {
          rc
        }
        activeMeetings: raceMeetings {
          id
          venueCode
          date
          status
          races {
            no
            postTime
            status
            wageringFieldSize
          }
        }
        raceMeetings(date: $date, venueCode: $venueCode) {
          id
          status
          venueCode
          date
          totalNumberOfRace
          currentNumberOfRace
          dateOfWeek
          meetingType
          totalInvestment
          country {
            code
            namech
            nameen
            seq
          }
          races {
            ...raceFragment
            runners {
              id
              no
              standbyNo
              status
              name_ch
              name_en
              horse {
                id
                code
              }
              color
              barrierDrawNumber
              handicapWeight
              currentWeight
              currentRating
              internationalRating
              gearInfo
              racingColorFileName
              allowance
              trainerPreference
              last6run
              saddleClothNo
              trumpCard
              priority
              finalPosition
              deadHeat
              winOdds
              jockey {
                code
                name_en
                name_ch
              }
              trainer {
                code
                name_en
                name_ch
              }
            }
          }
          obSt: pmPools(oddsTypes: [WIN, PLA]) {
            leg {
              races
            }
            oddsType
            comingleStatus
          }
          poolInvs: pmPools(
            oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
          ) {
            id
            leg {
              number
              races
            }
            status
            sellStatus
            oddsType
            investment
            mergedPoolId
            lastUpdateTime
          }
          ...racingBlockFragment
          pmPools(oddsTypes: []) {
            id
          }
          jkcInstNo: foPools(oddsTypes: [JKC], filters: ["top"]) {
            instNo
          }
          tncInstNo: foPools(oddsTypes: [TNC], filters: ["top"]) {
            instNo
          }
        }
      }
      """
  }

  # Make a POST request to the API
  response = requests.post(url, json=payload)

  # Check if the request was successful
  if response.status_code == 200:
      data = response.json()
      # Extract the 'race_no' and 'name_ch' fields and save them into a dictionary
      race_meetings = data.get('data', {}).get('raceMeetings', [])
      race_dict = {}
      post_time_dict = {}
      for meeting in race_meetings:
          for race in meeting.get('races', []):
              id = race.get('runners', [])[0].get('id')
              if id[8:10] != place:
                    continue
              race_number = race["no"]
              post_time = race.get("postTime", "Field not found")
              time_part = datetime.fromisoformat(post_time)
              post_time_dict[race_number] = time_part
              race_dict[race_number] = {"馬名": [], "騎師": [],'練馬師':[],'最近賽績':[]}
              for runner in race.get('runners', []):
                if runner.get('standbyNo') == "":
                  name_ch = runner.get('name_ch', 'Field not found')
                  jockey_name_ch = runner.get('jockey', {}).get('name_ch', 'Field not found')
                  trainer_name_ch = runner.get('trainer', {}).get('name_ch', 'Field not found')
                  last6run = runner.get('last6run', 'Field not found')
                  race_dict[race_number]["馬名"].append(name_ch)
                  race_dict[race_number]["騎師"].append(jockey_name_ch)
                  race_dict[race_number]["練馬師"].append(trainer_name_ch)
                  race_dict[race_number]["最近賽績"].append(last6run)
      print('完成')

  else:
      print(f'Failed to retrieve data. Status code: {response.status_code}')

  race_dataframes = {}
  numbered_dict ={}
  for race_number in race_dict:
      df = pd.DataFrame(race_dict[race_number])
      df.index += 1  # Set index to start from 1
      numbered_list = [f"{i+1}. {name}" for i, name in enumerate(race_dict[race_number]['馬名'])]
      numbered_dict[race_number] = numbered_list
      race_dataframes[race_number] = df
  
top_container = st.container()
placeholder = st.empty()
ucb_placeholder = st.empty()
if st.session_state.get('reset', False):
    with top_container:
      st.write(f"DataFrame for Race No: {race_no}")
      st.dataframe(race_dataframes[race_no], use_container_width=True)
    odds_dict = {}
    for method in methodlist:
        odds_dict[method] = pd.DataFrame()
    investment_dict = {}
    for method in methodlist:
        investment_dict[method] = pd.DataFrame()
    overall_investment_dict = {}
    for method in methodlist:
        overall_investment_dict.setdefault(method, pd.DataFrame())
    overall_investment_dict.setdefault('overall', pd.DataFrame())
    weird_dict = {}
    for method in methodlist:
        weird_dict.setdefault(method, pd.DataFrame([], columns=['No.', 'error', 'odds', 'Highlight']))
    diff_dict = {}
    for method in methodlist:
        diff_dict.setdefault(method, pd.DataFrame())
    diff_dict.setdefault('overall', pd.DataFrame())
    # 初始化狀態
    st.session_state.update({
        'ucb_state': {
            't': 0,
            'selected_count': {i+1: 0 for i in range(len(race_dict[race_no]['馬名']))},
            'momentum_hist': {i+1: [] for i in range(len(race_dict[race_no]['馬名']))},
            'final_ucb_df': None,
            'locked': False
        }
    })


    start_time = time.time()
    end_time = start_time + 60*10000

    while time.time()<=end_time:
        with placeholder.container():
            time_now = datetime.now() + datere.relativedelta(hours=8)
            odds = get_odds_data()
            investments = get_investment_data()

            # --- 你的原始分析 ---
            main(time_now, odds, investments, period=2)

            # --- UCB 預測 ---
            if 'WIN' in odds and odds['WIN']:
                win_odds = np.array([o if o != np.inf else 999 for o in odds['WIN']])
                total_win_inv = investments.get('WIN', [0])[0] if investments.get('WIN') else 0
                horses = list(range(1, len(win_odds) + 1))
                ucb_state = st.session_state.ucb_state
                ucb_state['t'] += 1

                df_ucb, top4 = run_ucb_prediction(win_odds, total_win_inv, horses, ucb_state, ucb_state['t'])

                # 開賽前 1 分鐘鎖定
                race_time = np.datetime64(post_time_dict[race_no]  + timedelta(hours=8))
                if (time_now - race_time).total_seconds() <= 60 and not ucb_state['locked']:
                    ucb_state['locked'] = True
                    ucb_state['final_ucb_df'] = df_ucb

                with ucb_placeholder.container():
                    st.subheader("UCB 即時預測（每次選 Top 4）")
                    if ucb_state['locked']:
                        st.success("最終預測鎖定！")
                        styled = df_ucb.style.apply(lambda row: ['background-color: lightgreen' if 'Top 1' in row['排名'] else
                                                              'background-color: lightyellow' if 'Top 2' in row['排名'] else
                                                              'background-color: lightsalmon' if 'Top 3' in row['排名'] else
                                                              'background-color: lightblue' if 'Top 4' in row['排名'] else '' for _ in row], axis=1)
                        st.dataframe(styled, use_container_width=True)
                        st.info(f"**最終 Top 4**：{', '.join([f'馬 {h}' for h in top4])}")
                    else:
                        st.dataframe(df_ucb.head(8), use_container_width=True)

            time.sleep(15)
            st.rerun()
