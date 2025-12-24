#!/usr/bin/env python3
import torch
import random
import re
import json
from transformers import AutoModelForMaskedLM, AutoTokenizer


# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese").to(device)
model.eval()

# 处理文本
def process_text(text):
    return re.sub(r'\s+', '', text).strip()

# 在输入的句子中插入掩码
def insert_mask(text):
    # 分割文本
    tokens = tokenizer.tokenize(text)
    # 随机选择插入位置,避免开头结尾
    insert_pos = random.randint(1, len(tokens)-1)
    tokens.insert(insert_pos, tokenizer.mask_token)

    # 生成带有mask的文本
    return tokenizer.convert_tokens_to_string(tokens)

def generate_text(text, max_length=100, num_iter=50):
    text = process_text(text)
    # 检查输入token数量
    if not text.strip():
        print('输入文本为空！')
        return ''

    current_tokens = tokenizer.tokenize(text)
    if len(current_tokens) >= max_length:
        print(f'文本长度{text}超过最大长度{max_length}!')
        return text

    # 迭代生成文本
    for _ in range(num_iter):
        text = insert_mask(text)

        # 编码文本
        inputs = tokenizer(text,
                           return_tensors='pt',
                           max_length=512,
                           truncation=True).to(device)

        # 获取输出
        outputs = model(**inputs)

        # 找到所有mask的索引
        mask_pos = [idx for idx, token_id in enumerate(inputs['input_ids'][0])
                   if token_id == tokenizer.mask_token_id]
        if not mask_pos:
            break

        # 遍历每个mask位置进行替换
        result_text = text
        for pos in mask_pos:
            # 提取logits
            preds = outputs.logits[0, pos]
            # 提取最大值
            max_pred = torch.argmax(preds).item()
            # 解码
            pred_token = tokenizer.decode([max_pred], skip_special_tokens=True)
            # 替换mask
            result_text = result_text.replace(tokenizer.mask_token, pred_token, 1)
        # 更新文本
        text = result_text
        # 计算更新后token数量
        current_tokens = tokenizer.tokenize(text)
        if len(current_tokens) >= max_length:
            break

    return process_text(text)

def clean_model_output(raw_output):
    # 定义清除模式：匹配从开头到最后一个"</think>"标签(含换行)之前的内容
    clean_pattern = r'^.*?<\/think>\s*'

    # 执行清除并保留核心内容
    cleaned_output = re.sub(clean_pattern, '', raw_output, flags=re.DOTALL)

    # 移除残留的调试标记
    cleaned_output = cleaned_output.replace('<think>', '').replace('</think>', '')

    # 去除换行符
    cleaned_output = cleaned_output.replace('\n', '')

    # 返回首尾去空格的纯净文本
    return cleaned_output.strip()

with open(r"C:\Users\1\Desktop\工作\dataset\中文数据集\原始数据\人类\test_data1-3\test.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取 inputtext 字段
texts = []
for item in data:
    for sub_item in item['texts']:  # 遍历 item 中的 texts 子列表
        texts.append(sub_item['inputtext'])

# 清空 data 以释放内存
data = []
rewritten_texts = []

for i in texts:
    rewritten_texts.append(generate_text(i, max_length=512, num_iter=50))
    print(f"已生成重写文本{len(rewritten_texts)}/{len(texts)}")

output_data = []
for original, rewritten in zip(texts, rewritten_texts):
    new_item = {
        "original_text": original,  # 原始文本
        "rewritten_text": rewritten,  # AI 重写文本
        "AI_percent": 1  # AI 生成标记
    }
    output_data.append(new_item)

with open(r"C:\Users\1\Desktop\工作\dataset\中文数据集\原始数据\人类\test_data1-3\test_output.json", 'w', encoding='utf-8') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=2)

print(f"已成功生成并保存{len(rewritten_texts)}条重写文本")













