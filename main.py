import os
import time
from datetime import datetime
import pandas as pd
import akshare as ak
import mplfinance as mpf
from openai import OpenAI

# ==========================================
# 1. 数据获取模块 (使用东方财富接口)
# ==========================================

def fetch_a_share_minute(symbol: str) -> pd.DataFrame:
    """
    获取A股1分钟K线 (使用东方财富接口，确保包含当日数据)
    """
    # 提取纯数字代码
    symbol_code = ''.join(filter(str.isdigit, symbol))
    
    print(f"正在获取 {symbol_code} 的1分钟数据 (Source: Eastmoney)...")

    try:
        # stock_zh_a_hist_min_em 是目前akshare获取分钟线最稳定的接口
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol_code, 
            period="1", 
            adjust="qfq"
        )
    except Exception as e:
        print(f"获取失败，请检查股票代码或网络: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # 东方财富接口返回的是中文列名，需要重新映射
    rename_map = {
        "时间": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # 格式转换
    df["date"] = pd.to_datetime(df["date"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    
    # 环境变量控制 K 线数量 (默认为 600)
    bars_count = int(os.getenv("BARS_COUNT", 600))
    df = df.sort_values("date").tail(bars_count).reset_index(drop=True)
    
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加威科夫分析所需的背景均线 (MA50/200)"""
    df = df.copy()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    return df

# ==========================================
# 2. 本地绘图模块
# ==========================================

def generate_local_chart(symbol: str, df: pd.DataFrame, save_path: str):
    """
    使用 mplfinance 在本地生成威科夫风格图表
    """
    if df.empty:
        return

    plot_df = df.copy()
    plot_df.set_index("date", inplace=True)

    # 设置样式
    mc = mpf.make_marketcolors(up='red', down='green', edge='i', wick='i', volume='in', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)

    # 添加均线图层
    apds = []
    if 'ma50' in plot_df.columns:
        apds.append(mpf.make_addplot(plot_df['ma50'], color='orange', width=1.0))
    if 'ma200' in plot_df.columns:
        apds.append(mpf.make_addplot(plot_df['ma200'], color='blue', width=1.2))

    try:
        mpf.plot(
            plot_df,
            type='candle',
            style=s,
            addplot=apds,
            volume=True,
            title=f"Wyckoff Chart: {symbol} (1-Min, Last {len(df)} Bars)",
            savefig=dict(fname=save_path, dpi=150, bbox_inches='tight'),
            warn_too_much_data=2000
        )
        print(f"[OK] Chart saved to: {save_path}")
    except Exception as e:
        print(f"[Error] 绘图失败: {e}")

# ==========================================
# 3. AI 分析模块 (从环境变量读取 Prompt)
# ==========================================

def ai_analyze_wyckoff(symbol: str, df: pd.DataFrame) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ai_model = os.getenv("AI_MODEL", "gpt-4o")

    if not api_key:
        return "错误：未设置 OPENAI_API_KEY。"

    # === 关键修正：从环境变量读取 Prompt 模板 ===
    prompt_template = os.getenv("WYCKOFF_PROMPT_TEMPLATE")
    
    # 本地调试时的回退逻辑 (可选：读取本地文件)
    if not prompt_template and os.path.exists("prompt_secret.txt"):
        try:
            with open("prompt_secret.txt", "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except:
            pass

    if not prompt_template:
        print("[Error] 未找到 WYCKOFF_PROMPT_TEMPLATE 环境变量。")
        return "错误：AI 分析提示词未配置，无法生成报告。"

    csv_data = df.to_csv(index=False)
    latest_price = df.iloc[-1]["close"]
    latest_time = df.iloc[-1]["date"]

    # 填充模板
    prompt = prompt_template.replace("{symbol}", symbol) \
                            .replace("{latest_time}", str(latest_time)) \
                            .replace("{latest_price}", str(latest_price)) \
                            .replace("{csv_data}", csv_data)

    print(f"正在请求 AI 分析 ({ai_model})...")
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        resp = client.chat.completions.create(
            model=ai_model, 
            messages=[
                {"role": "system", "content": "You are Richard D. Wyckoff. You follow strict Wyckoff logic."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 
        )
        return resp.choices[0].message.content
        
    except Exception as e:
        error_msg = f"""
# 分析服务异常
**错误详情**: `{str(e)}`
"""
        print(f"[Error] AI 调用失败: {e}")
        return error_msg

# ==========================================
# 4. 主程序
# ==========================================

def main():
    symbol = os.getenv("SYMBOL", "600970") 
    
    # 1. 获取数据
    df = fetch_a_share_minute(symbol)
    if df.empty:
        print("未获取到数据，程序终止。")
        return
        
    df = add_indicators(df)

    # 建立输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 2. 保存CSV
    csv_path = f"data/{symbol}_1min_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV Saved: {csv_path} ({len(df)} rows)")

    # 3. 本地生成图表
    chart_path = f"reports/{symbol}_chart_{ts}.png"
    generate_local_chart(symbol, df, chart_path)

    # 4. 生成威科夫分析报告
    report_text = ai_analyze_wyckoff(symbol, df)

    # 5. 保存报告
    report_path = f"reports/{symbol}_report_{ts}.md"
    
    # 将图片链接插入 Markdown 报告顶部
    final_report = f"![Wyckoff Chart](./{os.path.basename(chart_path)})\n\n{report_text}"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"[OK] Report Saved: {report_path}")

if __name__ == "__main__":
    main()
