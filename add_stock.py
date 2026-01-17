import os
import re
import time
import requests

def get_telegram_updates(bot_token):
    """获取 Telegram 机器人最近收到的消息"""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        # timeout=10 避免卡死
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("result", [])
    except Exception as e:
        print(f"获取消息失败: {e}")
    return []

def send_reply(bot_token, chat_id, text):
    """发送回复消息"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=data, timeout=5)
    except:
        pass

def main():
    bot_token = os.getenv("TG_BOT_TOKEN")
    admin_chat_id = os.getenv("TG_CHAT_ID")

    if not bot_token:
        print("未设置 TG_BOT_TOKEN")
        return

    # 1. 获取消息
    updates = get_telegram_updates(bot_token)
    if not updates:
        print("没有新消息")
        return

    # 2. 读取现有股票列表
    file_path = "stock_list.txt"
    existing_stocks = set()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_stocks = {line.strip() for line in f if line.strip()}

    new_stocks = set()
    latest_update_id = 0
    
    # 状态标记
    should_clear = False 
    should_view = False # === 新增：是否触发查看 ===

    # 3. 解析消息
    current_time = time.time()
    
    for update in updates:
        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "")
        date = message.get("date", 0)
        update_id = update.get("update_id")

        latest_update_id = max(latest_update_id, update_id)

        # 安全检查
        if admin_chat_id and chat_id != str(admin_chat_id):
            continue

        # 时间检查 (40分钟内)
        if current_time - date > 2400: 
            continue

        # === 指令 1: 检查查看/查询 ===
        if re.search(r"(查看|查询|列表|list|ls|cx)", text, re.IGNORECASE):
            should_view = True
            print(f"收到查看指令: '{text}'")

        # === 指令 2: 检查清空 ===
        if re.search(r"(清空|clear)", text, re.IGNORECASE):
            should_clear = True
            print(f"收到清空指令: '{text}'")

        # === 指令 3: 提取股票代码 ===
        codes = re.findall(r"\d{6}", text)
        for code in codes:
            new_stocks.add(code)
            print(f"发现股票代码: {code}")

    # 4. 处理变更 (清空 或 添加)
    list_changed = False
    
    if new_stocks or should_clear:
        list_changed = True
        final_list = set()
        
        if should_clear:
            # 清空后，只保留本次新增
            final_list = new_stocks
            action_msg = "🗑 <b>列表已清空。</b>"
        else:
            # 追加模式
            final_list = existing_stocks.union(new_stocks)
            action_msg = "✅ <b>已添加监控。</b>"

        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            for stock in sorted(final_list):
                f.write(f"{stock}\n")
        
        # 更新内存中的列表，以便后续"查看"使用最新数据
        existing_stocks = final_list
        
        # 发送变更通知
        if new_stocks:
            stock_str = ", ".join(sorted(new_stocks))
            msg = f"{action_msg}\n本次变动: {stock_str}"
        else:
            msg = f"{action_msg}"
        send_reply(bot_token, admin_chat_id, msg)

    # 5. 处理查看 (如果触发了查看，或者没有变动但有消息交互，反馈一下)
    # 逻辑：如果用户发了"查看"，或者单纯想确认，就发完整列表
    if should_view:
        if existing_stocks:
            # 格式化列表：每行一个，或者用逗号隔开
            sorted_list = sorted(existing_stocks)
            # 为了美观，每行显示 3 个，或者直接列表
            list_str = "\n".join([f"• <code>{code}</code>" for code in sorted_list])
            
            view_msg = (
                f"📋 <b>当前监控列表 ({len(sorted_list)}只):</b>\n"
                f"{list_str}"
            )
        else:
            view_msg = "📭 <b>当前监控列表为空。</b>"
            
        send_reply(bot_token, admin_chat_id, view_msg)

    # 6. 消费消息 (防止循环处理)
    if latest_update_id > 0:
        try:
            requests.get(f"https://api.telegram.org/bot{bot_token}/getUpdates?offset={latest_update_id + 1}", timeout=5)
        except:
            pass
        
    if not (list_changed or should_view):
        print("无有效指令。")

if __name__ == "__main__":
    main()
