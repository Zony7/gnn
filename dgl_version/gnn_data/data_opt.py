import pandas as pd

def extract_and_deduplicate_fault_info(input_file, output_file):
    """
    将输入Excel文件中的所有单元格内容（从第二行开始）提取并去重，
    然后将去重后的内容按顺序写入到输出Excel文件的第一列，并增加一列从0开始编号。

    参数:
    input_file (str): 输入Excel文件的路径。
    output_file (str): 输出Excel文件的路径。
    """
    # 读取输入文件
    df = pd.read_excel(input_file, header=None)

    # 初始化一个空列表来存储所有单元格内容
    all_cells = []

    # 遍历每一列，从第二行开始提取内容
    for col in df.columns:
        column_data = df[col].dropna().tolist()[1:]  # 去除 NaN 值并从第二行开始提取
        all_cells.extend(column_data)

    # 去重内容
    seen = set()
    unique_cells = []
    for cell in all_cells:
        if cell not in seen:
            unique_cells.append(cell)
            seen.add(cell)

    # 创建一个新的 DataFrame，将所有内容放在第一列，并增加一列从0开始编号
    new_df = pd.DataFrame({'Fault Info': unique_cells})
    new_df['Index'] = range(len(new_df))  # 增加一列从0开始编号

    # 将新的 DataFrame 写入到输出文件
    new_df.to_excel(output_file, index=False, header=False)


def generate_fault_knowledge_triples(input_file, output_file, description_col, relation_str, cause_col):
    """
    通过输入Excel文件生成故障知识三元组，并将结果保存到输出Excel文件中。
    第一列是Description，第二列是固定的字符串`原因是`，第三列是Cause。

    参数:
    input_file (str): 输入Excel文件的路径。
    output_file (str): 输出Excel文件的路径。
    description_col (str): 包含Description的列名。
    relation_str (str): 固定的关系字符串。
    cause_col (str): 包含Cause的列名。
    """
    # 读取输入文件
    df = pd.read_excel(input_file, header=0)  # 假设第一行是列名

    # 初始化一个空列表来存储三元组
    triples = []

    # 提取Description和Cause列的内容
    for index, row in df.iterrows():
        description = row[description_col]
        cause = row[cause_col]
        triples.append([description, relation_str, cause])

    # 创建一个新的 DataFrame
    new_df = pd.DataFrame(triples, columns=[description_col, 'Relation', cause_col])

    # 将新的 DataFrame 写入到输出文件，header=False 表示不写入列名
    new_df.to_excel(output_file, index=False, header=False)

def merge_fault_knowledge_files(output_file, *files):
    """
    合并多个Excel文件，并将结果保存到输出Excel文件中。
    简单地将多个文件的内容按行拼接在一起。

    参数:
    output_file (str): 输出Excel文件的路径。
    *files (str): 多个Excel文件的路径。
    """
    # 初始化一个空的 DataFrame 列表
    dataframes = []

    # 读取每个文件并添加到 DataFrame 列表中
    for file in files:
        df = pd.read_excel(file, header=None)
        dataframes.append(df)

    # 简单地将所有 DataFrame 按行拼接在一起
    merged_df = pd.concat(dataframes, ignore_index=True)

    # 将合并后的 DataFrame 写入到输出文件，header=False 表示不写入列名
    merged_df.to_excel(output_file, index=False, header=False)

def process_fault_knowledge(input_entity_file, input_fault_knowledge_file, output_file):
    # 读取故障实体.xlsx文件
    entity_df = pd.read_excel(input_entity_file)
    # 读取FaultKnowledge.xlsx文件
    fault_knowledge_df = pd.read_excel(input_fault_knowledge_file)
    # 创建一个映射字典，将故障实体.xlsx的第一列映射到第二列
    mapping_dict = dict(zip(entity_df.iloc[:, 0], entity_df.iloc[:, 1]))
    # 向字典中添加额外的映射关系
    mapping_dict['原因是'] = '1'
    mapping_dict['维修建议'] = '2'
    # 根据映射字典替换FaultKnowledge.xlsx的第一列
    fault_knowledge_df.iloc[:, 0] = fault_knowledge_df.iloc[:, 0].map(mapping_dict)
    # 根据映射字典替换FaultKnowledge.xlsx的第二列
    fault_knowledge_df.iloc[:, 1] = fault_knowledge_df.iloc[:, 1].map(mapping_dict)
    # 根据映射字典替换FaultKnowledge.xlsx的第三列
    fault_knowledge_df.iloc[:, 2] = fault_knowledge_df.iloc[:, 2].map(mapping_dict)
    # 设置新的列名
    fault_knowledge_df.columns = ['0', '1', '341']
    # 保存修改后的FaultKnowledge.xlsx文件
    fault_knowledge_df.to_excel(output_file, index=False)


def generate_test_set(input_fault_knowledge_file, output_file):
    # 读取FaultKnowledge.xlsx文件
    fault_knowledge_df = pd.read_excel(input_fault_knowledge_file, header=None)

    # 初始化一个空列表来存储键值对
    key_value_list = []

    # 用于存储已经遇到的键
    seen_keys = set()

    # 遍历每一行，提取第一列和第二列作为键值对
    for index, row in fault_knowledge_df.iterrows():
        if len(row) >= 2:  # 确保每行至少有两列
            key = row[0]
            value = row[1]
            if key in seen_keys:
                # 如果键已经存在，找到对应的键值对并添加到列表中
                for kv in key_value_list:
                    if kv[0] == key:
                        key_value_list.append(kv)
                        break
            else:
                # 如果键不存在，添加新的键值对
                key_value_list.append([key, value])
                seen_keys.add(key)

    # 将列表转换为DataFrame并保存到Excel文件
    map_df = pd.DataFrame(key_value_list, columns=['Key', 'Value'])
    map_df.to_excel(output_file, index=False, header=False)




def add_alternating_rows(input_file, output_file):
    """
    对于输入Excel文件的所有数据，每隔7个数据增加一行，
    这行和上一行的第一列和第二列相同，第三列如果上一行第三列为1那就改为0，如果上一行为0那就改为1。

    参数:
    input_file (str): 输入Excel文件的路径。
    output_file (str): 输出Excel文件的路径。
    """
    # 读取输入文件
    df = pd.read_excel(input_file, header=None)

    # 初始化一个空列表来存储新的行
    new_rows = []

    # 遍历每一行
    for i, row in df.iterrows():
        new_rows.append(row)
        if i <= 300 and i % 3 == 1 and i + 1 < len(df):
            new_row = row.copy()
            new_row[2] = 1 - new_row[2]
            new_rows.append(new_row)
        # else:
        #     new_rows.append(row)

    # 创建一个新的 DataFrame
    new_df = pd.DataFrame(new_rows)

    # 将新的 DataFrame 写入到输出文件
    new_df.to_excel(output_file, index=False, header=False)

def convert_xlsx_to_tsv(input_file, output_file):
    """
    将指定的xlsx文件转化为tsv文件。

    参数:
    input_file (str): 输入Excel文件的路径。
    output_file (str): 输出TSV文件的路径。
    """
    # 读取输入文件
    df = pd.read_excel(input_file, header=None)

    # 将 DataFrame 写入到输出文件，使用制表符分隔
    df.to_csv(output_file, sep='\t', index=False, header=False)

def duplicate_xlsx_file(input_file, output_file, n):
    """
    将输入的xlsx文件内容重复n次，并将结果保存到输出xlsx文件中。

    参数:
    input_file (str): 输入Excel文件的路径。
    output_file (str): 输出Excel文件的路径。
    n (int): 重复次数。
    """
    # 读取输入文件
    df = pd.read_excel(input_file, header=None)
    print(f"Read {len(df)} rows from {input_file}")  # 调试信息：打印读取的行数

    # 重复内容n次
    duplicated_df = pd.concat([df] * n, ignore_index=True)
    print(f"Duplicated to {len(duplicated_df)} rows")  # 调试信息：打印重复后的行数

    # 将重复后的内容写入到输出文件
    duplicated_df.to_excel(output_file, index=False, header=False)
    print(f"Written to {output_file}")  # 调试信息：打印写入的文件路径

def duplicate_tsv_file(input_file, output_file, n):
    """
    将输入的tsv文件内容重复n次，并将结果保存到输出tsv文件中。

    参数:
    input_file (str): 输入TSV文件的路径。
    output_file (str): 输出TSV文件的路径。
    n (int): 重复次数。
    """
    # 读取输入文件
    df = pd.read_csv(input_file, sep='\t', header=None, encoding='gbk')


    # 重复内容n次
    duplicated_df = pd.concat([df] * n, ignore_index=True)

    # 将重复后的内容写入到输出文件
    duplicated_df.to_csv(output_file, sep='\t', index=False, header=False)

def shuffle_rows_in_xlsx(input_file, output_file):
    """
    打乱输入Excel文件中行的顺序，并将结果保存到输出Excel文件中。

    参数:
    input_file (str): 输入Excel文件的路径。
    output_file (str): 输出Excel文件的路径。
    """
    # 读取输入文件
    df = pd.read_excel(input_file, header=None)

    # 打乱行的顺序
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # 将打乱后的 DataFrame 写入到输出文件
    shuffled_df.to_excel(output_file, index=False, header=False)


def train_opt(entity_file, fault_knowledge_file, output_file):
    # 读取故障实体.xlsx文件
    entity_df = pd.read_excel(entity_file)
    # 读取FaultKnowledge.xlsx文件
    fault_knowledge_df = pd.read_excel(fault_knowledge_file)
    # 创建一个映射字典，将故障实体.xlsx的第一列映射到第二列
    mapping_dict = dict(zip(entity_df.iloc[:, 0], entity_df.iloc[:, 1]))
    # 向字典中添加额外的映射关系
    mapping_dict['原因是'] = '1'
    mapping_dict['维修建议'] = '2'
    # 根据映射字典替换FaultKnowledge.xlsx的第一列
    fault_knowledge_df.iloc[:, 0] = fault_knowledge_df.iloc[:, 0].map(mapping_dict)
    # 根据映射字典替换FaultKnowledge.xlsx的第二列
    fault_knowledge_df.iloc[:, 1] = fault_knowledge_df.iloc[:, 1].map(mapping_dict)
    # 根据映射字典替换FaultKnowledge.xlsx的第三列
    fault_knowledge_df.iloc[:, 2] = fault_knowledge_df.iloc[:, 2].map(mapping_dict)
    # fault_knowledge_df.columns = ['0', '1', '322']
    # 增加新列，根据第二列的值映射成10086或10087
    fault_knowledge_df['1'] = fault_knowledge_df.iloc[:, 1].map({'1': '1', '2': '0'})
    # 删除第二列
    fault_knowledge_df.drop(columns=['原因是'], inplace=True)
    fault_knowledge_df.columns = ['0', '341', '1']
    fault_knowledge_df.to_excel(output_file, index=False)


def construct_fault_description(input_file, output_file):
    # 读取输入文件
    df = pd.read_excel(input_file, header=None)

    # 打印前几行数据，检查内容
    print("前几行数据 (原始):")
    print(df.head())

    # 处理每一行数据
    processed_data = []
    for index, row in df.iterrows():
        text = row[0]
        second_column = row[1] if len(row) > 1 else None

        # 在每行第一列的文字前面加上 "在网络设备中出现"
        processed_text = f"在网络设备中出现{text}"

        # 如果文本以句号结尾，去掉句号并在文本最后加上 "，故障原因是什么？"
        if processed_text.endswith('。'):
            processed_text = processed_text[:-1] + "，故障原因是什么？"
        elif processed_text.endswith('。 '):
            processed_text = processed_text[:-2] + "，故障原因是什么？"
        else:
            processed_text += "，故障原因是什么？"

        # 添加处理后的文本和第二列数据到列表
        if second_column is not None:
            processed_data.append([processed_text, second_column])
        else:
            processed_data.append([processed_text])

    # 创建新的 DataFrame
    new_df = pd.DataFrame(processed_data)

    # 打印前几行数据，检查处理结果
    # print("前几行数据 (处理后):")
    # print(new_df.head())

    # 保存新的 DataFrame 到新的 Excel 文件
    new_df.to_excel(output_file, index=False, header=False)
    print(f"处理后的文件已保存到 {output_file}")




if __name__ == "__main__":

    # 生成故障实体
    input_file = 'data/故障信息.xlsx'
    output_file = 'data/故障实体.xlsx'
    extract_and_deduplicate_fault_info(input_file, output_file)

    # 生成故障知识三元组
    input_file = 'data/故障信息.xlsx'
    output_file1 = 'data/Description-Cause.xlsx'
    generate_fault_knowledge_triples(input_file, output_file1, 'Description', '原因是', 'Cause')
    output_file2 = 'data/Cause-Suggestion.xlsx'
    generate_fault_knowledge_triples(input_file, output_file2, 'Cause', '维修建议', 'Suggestion')

    # 合并多个三元组文件
    file1 = output_file1
    file2 = output_file2
    output_file = 'data/FaultKnowledge.xlsx'
    merge_fault_knowledge_files(output_file, file1, file2)

    # 调用函数处理文件生成三元组
    process_fault_knowledge('data/故障实体.xlsx', 'data/FaultKnowledge.xlsx', 'data/三元组.xlsx')

    # 生成训练集
    train_opt('data/故障实体.xlsx', 'data/FaultKnowledge.xlsx', 'data/训练集.xlsx')


    # 增加噪声
    add_alternating_rows('data/训练集.xlsx', 'data/训练集1.xlsx')

    # 调用函数将xlsx文件内容重复n次
    duplicate_xlsx_file('data/训练集1.xlsx', 'data/训练集2.xlsx', 1)


    # 打乱xlsx文件中行的顺序
    shuffle_rows_in_xlsx('data/训练集2.xlsx', 'data/打乱后的训练集.xlsx')
    shuffle_rows_in_xlsx('data/三元组.xlsx', 'data/乱序三元组.xlsx')

    # 转化xlsx为tsv
    convert_xlsx_to_tsv('data/打乱后的训练集.xlsx', 'data/rating_index.tsv')
    convert_xlsx_to_tsv('data/乱序三元组.xlsx', 'data/kg_index.tsv')

    # 生成测试集合
    generate_test_set('data/故障信息.xlsx', 'data/测试集.xlsx')

    # 构造故障描述文本
    construct_fault_description('data/测试集.xlsx', 'data/故障描述文本.xlsx')