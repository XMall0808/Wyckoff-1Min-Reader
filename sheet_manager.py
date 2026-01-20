import os
import json
import gspread
from datetime import datetime

class SheetManager:
    def __init__(self):
        # 1. 读取环境变量
        json_str = os.getenv("GCP_SA_KEY")
        sheet_name = os.getenv("SHEET_NAME", "Wyckoff_Stock_List")
        
        if not json_str:
            raise ValueError("❌ 未找到 GCP_SA_KEY 环境变量，请检查 GitHub Secrets")

        # 2. 解析 JSON
        try:
            creds_dict = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("❌ GCP_SA_KEY 格式错误，请确保 Secret 里填的是完整的 JSON 内容")

        # 3. 连接 Google Sheets (使用新版原生方法)
        try:
            # 这一步会自动处理 Scope 和 Token
            self.client = gspread.service_account_from_dict(creds_dict)
            self.sheet = self.client.open(sheet_name).sheet1
        except Exception as e:
            # 捕获并打印详细错误
            error_msg = str(e)
            # 如果是 API 错误，尝试提取详情
            if hasattr(e, 'response') and e.response:
                try:
                    error_msg += f" | API返回: {e.response.text}"
                except:
                    pass
            raise Exception(f"GSpread 初始化失败: {error_msg}")

    def get_all_stocks(self):
        """读取所有股票，返回字典格式"""
        try:
            records = self.sheet.get_all_records()
        except Exception as e:
            # 如果表格为空或读取失败，打印警告但不崩溃
            print(f"⚠️ 读取表格记录失败 (可能是表格为空): {e}")
            return {}
        
        stocks = {}
        for row in records:
            # 兼容处理:有些时候 Key 可能是 'Code ' (带空格)
            code_key = next((k for k in row.keys() if 'Code' in str(k)), None)
            if not code_key: continue

            code = str(row[code_key]).strip()
            if code:
                # 容错处理：找不到列就给默认值
                date = str(row.get('BuyDate', '')).strip() or datetime.now().strftime("%Y-%m-%d")
                qty = str(row.get('Qty', '')).strip() or "0"
                price = str(row.get('Price', '')).strip() or "0.0"
                
                stocks[code] = {'date': date, 'qty': qty, 'price': price}
        return stocks

    def add_or_update_stock(self, code, date=None, qty=None, price=None):
        """添加或更新股票"""
        code = str(code)
        date = date or datetime.now().strftime("%Y-%m-%d")
        qty = qty or 0
        price = price or 0.0
        
        try:
            # 查找代码所在的单元格 (默认第一列)
            cell = self.sheet.find(code)
            # 存在则更新: Code(1), BuyDate(2), Qty(3), Price(4)
            self.sheet.update_cell(cell.row, 2, date)
            self.sheet.update_cell(cell.row, 3, qty)
            self.sheet.update_cell(cell.row, 4, price)
            return "Updated"
        except gspread.exceptions.CellNotFound:
            # 不存在则追加
            self.sheet.append_row([code, date, qty, price])
            return "Added"

    def remove_stock(self, code):
        """删除股票"""
        try:
            cell = self.sheet.find(str(code))
            self.sheet.delete_rows(cell.row)
            return True
        except gspread.exceptions.CellNotFound:
            return False

    def clear_all(self):
        """清空（保留表头）"""
        self.sheet.resize(rows=1) 
        self.sheet.resize(rows=100)
