import pandas as pd

def xlsx_to_csv(xlsx_path, csv_path):
    # 读取xlsx文件
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    # 将数据写入csv文件
    df.to_csv(csv_path, index=False)

# 使用示例
xlsx_path = "data/故障信息.xlsx"  # 假设你的xlsx文件名为“豆瓣电影Top250.xlsx”
csv_path = "data/故障信息.csv"    # 转换后的csv文件名为“豆瓣电影Top250.csv”
xlsx_to_csv(xlsx_path, csv_path)
print("转换完成！")
