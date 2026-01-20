import os
import time
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import akshare as ak
import mplfinance as mpf
from openai import OpenAI
import numpy as np
import markdown
from xhtml2pdf import pisa
from sheet_manager import SheetManager 
import concurrent.futures 

# ==========================================
# 1. 数据获取模块 (固定 500根 5min)
# ==========================================

def fetch_stock_data_dynamic(symbol: str, buy_date_str: str) -> dict:
    clean_digits = ''.join(filter(str.isdigit, str(symbol)))
    symbol_code = clean_digits.zfill(6)
    
    # === 核心修改：固定获取策略 ===
    # 5分钟K线，每天48根。500根大约需要 10.5 个交易日。
    # 为了保险（考虑周末、节假日），我们直接向前推 40 天，保证数据够多。
    start_date_em = (datetime.now() - timedelta(days=40)).strftime("%Y%m%d")

    # print(f"   -> 正在获取 {symbol_code} 5分钟数据 (Limit: 500)...")

    try:
        # 获取 5分钟 数据
        df = ak.stock_zh_a_hist_min_em(symbol=symbol_code, period="5", start_date=start_date_em, adjust="qfq")
    except Exception as e:
        print(f"   [Error] {symbol_code} AkShare接口报错: {e}")
        return {"df": pd.DataFrame(), "period": "5m"}

    if df.empty:
        return {"df": pd.DataFrame(), "period": "5m"}

    # === 数据清洗 ===
    rename_map = {"时间": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"])
    
    cols = ["open", "high", "low", "close", "volume"]
    valid_cols = [c for c in cols if c in df.columns]
    df[valid_cols] = df[valid_cols].astype(float)

    # 修复开盘价为0
    if "open" in df.columns and (df["open"] == 0).any():
        df["open"] = df["open"].replace(0, np.nan)
        if "close" in df.columns:
            df["open"] = df["open"].fillna(df["close"].shift(1)).fillna(df["close"])

    # === 核心修改：强制截取最后 500 根 ===
    if len(df) > 500:
        df = df.tail(500).reset_index(drop=True)
    
    return {"df": df, "period": "5m"}

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" in df.columns:
        df["ma50"] = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()
    return df

# ==========================================
# 2. 绘图模块
# ==========================================

def generate_local_chart(symbol: str, df: pd.DataFrame, save_path: str, period: str):
    if df.empty: return
    plot_df = df.copy()
    if "date" in plot_df.columns: plot_df.set_index("date", inplace=True)

    mc = mpf.make_marketcolors(up='#ff3333', down='#00b060', edge='inherit', wick='inherit', volume={'up': '#ff3333', 'down': '#00b060'}, inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridstyle=':', y_on_right=True)
    apds = []
    if 'ma50' in plot_df.columns: apds.append(mpf.make_addplot(plot_df['ma50'], color='#ff9900', width=1.5))
    if 'ma200' in plot_df.columns: apds.append(mpf.make_addplot(plot_df['ma200'], color='#2196f3', width=2.0))

    try:
        # title 增加显示 bar count
        mpf.plot(plot_df, type='candle', style=s, addplot=apds, volume=True, 
                 title=f"Wyckoff: {symbol} ({period} | {len(plot_df)} bars)", 
                 savefig=dict(fname=save_path, dpi=150, bbox_inches='tight'), 
                 warn_too_much_data=2000)
    except Exception as e:
        print(f"   [Error] {symbol} 绘图失败: {e}")

# ==========================================
# 3. AI 分析模块 (Fail Fast & Auto-Fallback)
# ==========================================

def get_prompt_content(symbol, df, position_info):
    prompt_template = os.getenv("WYCKOFF_PROMPT_TEMPLATE")
    if not prompt_template and os.path.exists("prompt_secret.txt"):
        try:
            with open("prompt_secret.txt", "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except: pass
    if not prompt_template: return None

    csv_data = df.to_csv(index=False)
    latest = df.iloc[-1]

    base_prompt = prompt_template.replace("{symbol}", symbol) \
                          .replace("{latest_time}", str(latest["date"])) \
                          .replace("{latest_price}", str(latest["close"])) \
                          .replace("{csv_data}", csv_data)
    
    def safe_get(key):
        val = position_info.get(key)
        if val is None or str(val).lower() == 'nan' or str(val).strip() == '':
            return 'N/A'
        return val

    buy_date = safe_get('date')
    buy_price = safe_get('price')
    qty = safe_get('qty')

    position_text = (
        f"\n\n[USER POSITION DATA]\n"
        f"Symbol: {symbol}\n"
        f"Buy Date: {buy_date}\n"
        f"Cost Price: {buy_price}\n"
        f"Quantity: {qty}\n"
        f"(Note: Please analyze the current trend based on this position data. If position data is N/A, analyze as a potential new entry.)"
    )
    
    return base_prompt + position_text

def call_gemini_http(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY missing")
    
    model_name = os.getenv("GEMINI_MODEL") 
    if not model_name: model_name = "gemini-1.5-flash"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "system_instruction": {"parts": [{"text": "You are Richard D. Wyckoff."}]},
        "generationConfig": {"temperature": 0.2},
        "safetySettings": safety_settings 
    }
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=180)
            
            if resp.status_code == 200:
                result = resp.json()
                if "error" in result: raise Exception(f"Logic Error: {result['error']}")
                
                candidates = result.get('candidates', [])
                if not candidates: raise ValueError("No candidates")
                
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if not parts: raise ValueError("Empty parts")
                
                text = parts[0].get('text', '')
                if not text: raise ValueError("Empty text")
                
                return text 
            
            # 429 Limit -> 直接切 OpenAI
            elif resp.status_code == 429:
                raise Exception(f"Gemini 429 Rate Limit: {resp.text[:100]}")

            # 503 Overload -> 重试
            elif resp.status_code == 503:
                print(f"   ⚠️ Gemini 503 Overloaded... Waiting 5s")
                time.sleep(5)
                continue
            
            else:
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")

        except Exception as e:
            if "429" in str(e): raise e
            if attempt == max_retries - 1:
                print(f"   ❌ Gemini Final Fail: {e}")
                raise e
            print(f"   ⚠️ Gemini Error (Attempt {attempt+1}): {e}... Retrying")
            time.sleep(2)
            
    raise Exception("Gemini Max Retries Exceeded")

def call_openai_official(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise ValueError("OpenAI Key missing")
    
    model_name = os.getenv("AI_MODEL", "gpt-4o")
    
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name, 
        messages=[{"role": "system", "content": "You are Richard D. Wyckoff."}, {"role": "user", "content": prompt}],
        temperature=0.2 
    )
    return resp.choices[0].message.content

def ai_analyze(symbol, df, position_info):
    prompt = get_prompt_content(symbol, df, position_info)
    if not prompt: return "Error: No Prompt"
    
    try: 
        return call_gemini_http(prompt)
    except Exception as e: 
        print(f"   ⚠️ [{symbol}] Gemini 失败 (转切 OpenAI): {str(e)[:80]}...")
        try: 
            return call_openai_official(prompt)
        except Exception as e2: 
            return f"Analysis Failed. Gemini Error: {e}. OpenAI Error: {e2}"

# ==========================================
# 4. PDF 生成模块
# ==========================================

def generate_pdf_report(symbol, chart_path, report_text, pdf_path):
    html_content = markdown.markdown(report_text)
    abs_chart_path = os.path.abspath(chart_path)
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if not os.path.exists(font_path): font_path = "msyh.ttc" 
    
    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @font-face {{ font-family: "MyChineseFont"; src: url("{font_path}"); }}
            @page {{ size: A4; margin: 1cm; }}
            body {{ font-family: "MyChineseFont", sans-serif; font-size: 12px; line-height: 1.5; }}
            h1, h2, h3, p, div {{ font-family: "MyChineseFont", sans-serif; color: #2c3e50; }}
            img {{ width: 18cm; margin-bottom: 20px; }}
            .header {{ text-align: center; margin-bottom: 20px; color: #7f8c8d; font-size: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">Wyckoff Quantitative Analysis | {symbol}</div>
        <img src="{abs_chart_path}" />
        <hr/>
        {html_content}
    </body>
    </html>
    """
    try:
        with open(pdf_path, "wb") as pdf_file:
            pisa.CreatePDF(full_html, dest=pdf_file)
        return True
    except: return False

# ==========================================
# 5. 主程序 (串行处理)
# ==========================================

def process_one_stock(symbol: str, position_info: dict):
    if position_info is None: position_info = {}
    clean_digits = ''.join(filter(str.isdigit, str(symbol)))
    clean_symbol = clean_digits.zfill(6)

    print(f"🚀 [{clean_symbol}] 开始分析...")

    data_res = fetch_stock_data_dynamic(clean_symbol, position_info.get('date'))
    df = data_res["df"]
    period = data_res["period"]
    
    if df.empty:
        print(f"   ⚠️ [{clean_symbol}] 数据为空，跳过")
        return None
    
    df = add_indicators(df)

    beijing_tz = timezone(timedelta(hours=8))
    ts = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    
    csv_path = f"data/{clean_symbol}_{period}_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    chart_path = f"reports/{clean_symbol}_chart_{ts}.png"
    pdf_path = f"reports/{clean_symbol}_report_{period}_{ts}.pdf"
    
    generate_local_chart(clean_symbol, df, chart_path, period)
    report_text = ai_analyze(clean_symbol, df, position_info)
    
    if generate_pdf_report(clean_symbol, chart_path, report_text, pdf_path):
        print(f"✅ [{clean_symbol}] 报告生成完毕")
        return pdf_path
    
    return None

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("☁️ 正在连接 Google Sheets...")
    try:
        sm = SheetManager()
        stocks_dict = sm.get_all_stocks()
        print(f"📋 获取 {len(stocks_dict)} 个任务")
    except Exception as e:
        print(f"❌ Sheet 连接失败: {e}")
        return

    generated_pdfs = []
    
    # 串行处理 (Max Workers = 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_symbol = {
            executor.submit(process_one_stock, symbol, info): symbol 
            for symbol, info in stocks_dict.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    generated_pdfs.append(result)
            except Exception as exc:
                print(f"❌ [{symbol}] 处理发生异常: {exc}")

    if generated_pdfs:
        print(f"\n📝 生成推送清单 ({len(generated_pdfs)}):")
        with open("push_list.txt", "w", encoding="utf-8") as f:
            for pdf in generated_pdfs:
                print(f"   -> {pdf}")
                f.write(f"{pdf}\n")
    else:
        print("\n⚠️ 无报告生成")

if __name__ == "__main__":
    main()
