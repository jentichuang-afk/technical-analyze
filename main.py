import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
from groq import Groq

# --- 1. 頁面設定 ---
st.set_page_config(page_title="股票大師：回測實驗室", layout="wide", page_icon="🧪")
st.title("🧪 股票大師：策略回測與獲利驗證")

# --- 安全性設定 ---
gemini_ok = False
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-flash-latest') 
    gemini_ok = True
except: pass

groq_ok = False
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=groq_key)
    groq_ok = True
except: pass

# --- 2. 側邊欄 ---
st.sidebar.header("⚙️ 參數設定")
ticker_input = st.sidebar.text_input("輸入股票代碼", value="2330", help="台股請輸入如 2330, 8155")
days_input = st.sidebar.slider("回測/觀察天數", 100, 1000, 365) # 增加天數上限以便回測

st.sidebar.subheader("💰 回測設定")
initial_capital = st.sidebar.number_input("初始資金", value=1000000)
strategy_type = st.sidebar.selectbox("選擇回測策略", ["均線策略 (MA5穿過MA20)", "KD策略 (低檔金叉/高檔死叉)"])

if st.sidebar.button("🔄 執行分析與回測"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.info("💡 提示：回測不包含手續費與滑價，僅供策略邏輯驗證。")

# --- 3. 核心數據處理 ---
@st.cache_data(ttl=300)
def get_stock_data(symbol, days):
    try:
        end_date = datetime.now() + timedelta(days=1)
        start_date = end_date - timedelta(days=days+150)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        return df
    except: return None

# --- 4. 技術指標計算 ---
def add_indicators(df):
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # MA
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    k_list = [50]; d_list = [50]
    for r in df['RSV']:
        if pd.isna(r): k_list.append(50); d_list.append(50)
        else:
            k = (2/3) * k_list[-1] + (1/3) * r
            d = (2/3) * d_list[-1] + (1/3) * k
            k_list.append(k); d_list.append(d)   
    df['K'] = k_list[1:]; df['D'] = d_list[1:]
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df

# --- 5. 🆕 回測引擎 (Backtest Engine) ---
def run_backtest(df, strategy, capital):
    df = df.copy().dropna()
    cash = capital
    position = 0 # 持有股數
    records = [] # 交易紀錄
    equity_curve = [] # 資產曲線
    
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []

    for i in range(1, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        prev_price = df['Close'].iloc[i-1]
        
        signal = 0 # 1=Buy, -1=Sell
        
        # --- 策略邏輯 ---
        if strategy == "均線策略 (MA5穿過MA20)":
            # 黃金交叉買進
            if df['MA5'].iloc[i] > df['MA20'].iloc[i] and df['MA5'].iloc[i-1] <= df['MA20'].iloc[i-1]:
                signal = 1
            # 死亡交叉賣出
            elif df['MA5'].iloc[i] < df['MA20'].iloc[i] and df['MA5'].iloc[i-1] >= df['MA20'].iloc[i-1]:
                signal = -1
                
        elif strategy == "KD策略 (低檔金叉/高檔死叉)":
            k_curr = df['K'].iloc[i]
            d_curr = df['D'].iloc[i]
            k_prev = df['K'].iloc[i-1]
            d_prev = df['D'].iloc[i-1]
            
            # K < 30 且 黃金交叉 -> 買
            if k_curr < 30 and k_curr > d_curr and k_prev <= d_prev:
                signal = 1
            # K > 80 且 死亡交叉 -> 賣
            elif k_curr > 80 and k_curr < d_curr and k_prev >= d_prev:
                signal = -1

        # --- 執行交易 ---
        if signal == 1 and position == 0: # 買進
            position = cash / price
            cash = 0
            buy_dates.append(date)
            buy_prices.append(price)
            records.append({"日期": date, "動作": "買進", "價格": price, "資產": position*price})
            
        elif signal == -1 and position > 0: # 賣出
            cash = position * price
            position = 0
            sell_dates.append(date)
            sell_prices.append(price)
            records.append({"日期": date, "動作": "賣出", "價格": price, "資產": cash})
            
        # 計算每日總資產
        current_equity = cash + (position * price)
        equity_curve.append(current_equity)

    # 績效統計
    final_equity = equity_curve[-1]
    total_return = (final_equity - capital) / capital * 100
    
    # 買入持有 (Buy & Hold) 績效
    bnh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "bnh_return": bnh_return,
        "equity_curve": equity_curve,
        "records": pd.DataFrame(records),
        "buy_points": (buy_dates, buy_prices),
        "sell_points": (sell_dates, sell_prices),
        "dates": df.index[1:]
    }

# --- 6. AI Prompt ---
def get_prompt(symbol, last_close, technical_data):
    return f"""
    角色：華爾街操盤手。標的：{symbol}，現價：{last_close:.2f}。
    
    【近5日技術數據】
    {technical_data}
    
    請進行純技術分析：
    1. 趨勢判讀 (均線排列)。
    2. 指標訊號 (KD, MACD)。
    3. 操作建議 (支撐/壓力)。
    """

def call_ai(model, prompt):
    try:
        if model == 'gemini' and gemini_ok:
            return gemini_model.generate_content(prompt).text
        elif model == 'groq' and groq_ok:
            return groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            ).choices[0].message.content
    except Exception as e: return f"AI 忙碌: {e}"
    return "API Key 未設定"

# --- 7. 主程式 ---
if ticker_input:
    raw_ticker = ticker_input.strip().upper()
    
    final_symbol = raw_ticker
    df = None
    
    with st.spinner(f"正在分析 {raw_ticker} ..."):
        if raw_ticker.isdigit():
            for s in ['.TW', '.TWO']:
                df = get_stock_data(raw_ticker + s, days_input)
                if df is not None:
                    final_symbol = raw_ticker + s
                    break
        else:
            df = get_stock_data(raw_ticker, days_input)
    
    if df is None:
        st.error(f"❌ 查無代碼 {raw_ticker}")
    else:
        df = add_indicators(df)
        df_display = df.iloc[-days_input:] # 只取回測天數
        last = df.iloc[-1]
        
        # 執行回測
        bt_result = run_backtest(df_display, strategy_type, initial_capital)
        
        st.markdown(f"## 🧪 {final_symbol} 回測報告 ({strategy_type})")
        
        # 回測結果看板
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("期末總資產", f"${bt_result['final_equity']:,.0f}")
        
        # 顏色邏輯
        ret_color = "normal"
        if bt_result['total_return'] > 0: ret_color = "off" # Streamlit metric delta logic workaround
        
        c2.metric("策略報酬率", f"{bt_result['total_return']:.2f}%", delta=f"{bt_result['total_return']:.2f}%")
        c3.metric("買進持有報酬", f"{bt_result['bnh_return']:.2f}%", help="如果第一天買進就不動，會賺多少")
        
        # 比較
        win_msg = "🏆 策略戰勝大盤！" if bt_result['total_return'] > bt_result['bnh_return'] else "🐢 輸給無腦買進"
        c4.write(f"### {win_msg}")

        # 分頁
        tab1, tab2, tab3 = st.tabs(["📊 買賣點圖表", "📝 交易明細", "🤖 AI 技術觀點"])
        
        with tab1:
            # 繪製回測圖
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            
            # K線
            fig.add_trace(go.Candlestick(x=df_display.index, open=df_display['Open'], high=df_display['High'], 
                                         low=df_display['Low'], close=df_display['Close'], name='K線'), row=1, col=1)
            
            # 均線
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA5'], line=dict(color='yellow', width=1), name='MA5'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
            
            # 買賣點標記
            buys = bt_result['buy_points']
            sells = bt_result['sell_points']
            
            fig.add_trace(go.Scatter(x=buys[0], y=buys[1], mode='markers', marker=dict(color='red', size=10, symbol='triangle-up'), name='買進訊號'), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells[0], y=sells[1], mode='markers', marker=dict(color='green', size=10, symbol='triangle-down'), name='賣出訊號'), row=1, col=1)
            
            # 資產曲線
            fig.add_trace(go.Scatter(x=bt_result['dates'], y=bt_result['equity_curve'], line=dict(color='cyan', width=2), name='資產曲線', fill='tozeroy'), row=2, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.dataframe(bt_result['records'])
            
        with tab3:
            # AI 分析部分
            target_cols = ['Close', 'MA5', 'MA20', 'MA60', 'K', 'D', 'MACD', 'MACD_Hist']
            tech_data_str = df.tail(5)[target_cols].to_string()
            prompt = get_prompt(final_symbol, last['Close'], tech_data_str)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Gemini")
                if gemini_ok: st.info(call_ai('gemini', prompt))
            with col2:
                st.subheader("Llama 3")
                if groq_ok: st.warning(call_ai('groq', prompt))
