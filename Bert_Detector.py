#!/usr/bin/env python3
import torch
import numpy as np
import json
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.cuda.amp import autocast

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model_ref = AutoModelForMaskedLM.from_pretrained("bert-base-chinese").to(device) # 参考模型
model_score = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext").to(device) # 评分模型
model_ref.eval()
model_score.eval()

# 计算采样差异
def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)  # 交叉熵
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)  # 方差
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / (var_ref.sum(dim=-1).sqrt() + 1e-10)  # 标准化差异
    discrepancy = discrepancy.mean()
    return discrepancy.item()

# 处理文本
def merge_paragraphs(data):
    paragraphs = re.split(r'\n', data)
    paragraphs = [para.strip() for para in paragraphs if para.strip()]  # 去除空段落
    temp_para = ''
    merged_paragraphs = []
    for para in paragraphs:
        if para[-1] in (":", ";", "：", "；"):
            temp_para += para
        else:
            temp_para += para
            merged_paragraphs.append(temp_para)
            temp_para = ''
    if temp_para:
        merged_paragraphs.append(temp_para)
    merged_paragraphs = [para for para in merged_paragraphs if para.strip()]  # 去除空段落
    return merged_paragraphs


def split_sentences(sentence):
    pattern = r'(?<=[，。！？：；、])(?!\d+)|(?<=[,.!?:;])(?=[A-Z]|$| |\n)'
    sentences = re.split(pattern, sentence)
    sentences = [s for s in sentences if s.strip()]
    chunk = 300

    step = len(sentence) // chunk + 1
    sentence_length = len(sentence) // step + 1
    end_length = 0.8 * sentence_length

    combined_paragraphs = []
    current_paragraph = ""

    for s in sentences:
        diff1 = sentence_length - len(current_paragraph)
        diff2 = len(current_paragraph) + len(s) - sentence_length
        if len(s) > sentence_length:
            combined_paragraphs.append(current_paragraph)

        elif diff1 < diff2:
            if len(current_paragraph) >= (end_length):
                combined_paragraphs.append(current_paragraph)
                current_paragraph = s
            else:
                current_paragraph += s

        elif diff1 > diff2:
            current_paragraph += s

    # 添加最后一个段落（如果不为空）
    if len(current_paragraph) >= end_length:
        combined_paragraphs.append(current_paragraph)
    else:
        combined_paragraphs[-1] += current_paragraph
        x = combined_paragraphs[-1]
    combined_paragraphs = [para for para in combined_paragraphs if para.strip()]  # 去除空段落
    return combined_paragraphs# 处理文本
def merge_paragraphs(data):
    paragraphs = re.split(r'\n', data)
    paragraphs = [para.strip() for para in paragraphs if para.strip()]  # 去除空段落
    temp_para = ''
    merged_paragraphs = []
    for para in paragraphs:
        if para[-1] in (":", ";", "：", "；"):
            temp_para += para
        else:
            temp_para += para
            merged_paragraphs.append(temp_para)
            temp_para = ''
    if temp_para:
        merged_paragraphs.append(temp_para)
    merged_paragraphs = [para for para in merged_paragraphs if para.strip()]  # 去除空段落
    return merged_paragraphs


def split_sentences(sentence):
    pattern = r'(?<=[，。！？：；、])(?!\d+)|(?<=[,.!?:;])(?=[A-Z]|$| |\n)'
    sentences = re.split(pattern, sentence)
    sentences = [s for s in sentences if s.strip()]
    chunk = 300

    step = len(sentence) // chunk + 1
    sentence_length = len(sentence) // step + 1
    end_length = 0.8 * sentence_length

    combined_paragraphs = []
    current_paragraph = ""

    for s in sentences:
        diff1 = sentence_length - len(current_paragraph)
        diff2 = len(current_paragraph) + len(s) - sentence_length
        if len(s) > sentence_length:
            combined_paragraphs.append(current_paragraph)

        elif diff1 < diff2:
            if len(current_paragraph) >= (end_length):
                combined_paragraphs.append(current_paragraph)
                current_paragraph = s
            else:
                current_paragraph += s

        elif diff1 > diff2:
            current_paragraph += s

    # 添加最后一个段落（如果不为空）
    if len(current_paragraph) >= end_length:
        combined_paragraphs.append(current_paragraph)
    else:
        combined_paragraphs[-1] += current_paragraph
        x = combined_paragraphs[-1]
    combined_paragraphs = [para for para in combined_paragraphs if para.strip()]  # 去除空段落
    return combined_paragraphs

# 计算文本分数
def getScore(sentence, max_length=512):
    if not sentence.strip() or len(sentence) < 1:
        return 0
    # 编码
    encodings = tokenizer(sentence,
                          return_tensors="pt",
                          truncation=True,
                          max_length=max_length)
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    if input_ids.size(1) < 3:
        return 0

    # 前向传播
    with torch.no_grad():
        with autocast():
            # 参考模型logits
            outputs_ref = model_ref(input_ids=input_ids, attention_mask=attention_mask)
            logits_ref = outputs_ref.logits[:, 1:-1, :] # 去掉开头结尾的特殊token
            # 评分模型的logits
            outputs_score = model_score(input_ids=input_ids, attention_mask=attention_mask)
            logits_score = outputs_score.logits[:, 1:-1, :]

            # 标签
            labels = input_ids[:, 1:-1] # 去掉开头结尾的特殊token

            # 计算差异
            discrepancy = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)

            # 使用sigmoid转换为概率
            prob = 1 / (1 + np.exp(-discrepancy))

    return prob

# 处理输入数据
def detect_text(input_file_path, output_file_path):
    # 设置阈值
    threshold_suspicious = 0.5 # 可疑文本
    threshold_ai = 0.6 # ai文本
    # 读取数据
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    results = []
    for item in data:
        for sub_item in item['texts']:  # 遍历 item 中的 texts 子列表
            paragraph = sub_item['inputtext']

            # 按照"\n"对段落进行分块，储存在paragraphs中,筛选字数大于150的段落
            S_vectors = []
            P_vectors = []
            paragraphs = merge_paragraphs(paragraph)

            AI_num = 0
            All_num = 0
            SES_num = 0

            for paragraph in paragraphs:
                # 将段落进行分句，调整为合适的长度
                sentence = re.sub(r"\[[0-9]+\]\【[0-9]+\】\①-⑨", "", paragraph)  # 去除类似 [1]【1】① 的引用符号
                if len(paragraph) < 30:
                    continue
                lines = split_sentences(sentence)
                ParaProbs = []
                ParaLines = []
                temp_sentence_vectors = []

                for line in lines:
                    if line == None:  # 如果句子中没有有效字符，则跳过
                        continue
                    prob = getScore(line)
                    prob = float(prob)
                    ParaLines.append(line)
                    ParaProbs.append(prob)
                    All_num += len(line)

                    if prob > threshold_ai:
                        label = "1"
                        AI_num += len(line)
                    elif prob > threshold_suspicious:
                        label = "2"
                        SES_num += len(line)
                    else:
                        label = "0"

                    temp_sentence_vectors.append([line, label, prob])

                S_vectors.append(temp_sentence_vectors)  # 生成句级向量
                ParaProb = sum(ParaProbs) / len(ParaProbs) if len(ParaProbs) != 0 else 0
                ParaLabel = "1" if ParaProb > threshold_ai else "0"
                P_vectors.append([ParaLines, ParaLabel, ParaProb])  # 添加段级向量

            final_prob = AI_num / All_num if All_num != 0 else 0
            final_label = "AI" if final_prob > threshold_ai else "正常" if final_prob < threshold_suspicious else "可疑的"
            final_vectors = P_vectors  # 选择最终的段级或句级向量
            results.append({
              "final_label": final_label,
              "final_prob": final_prob,
              "final_vectors": final_vectors})

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"已处理{len(results)}条文本并保存结果")


# 调用主函数
if __name__ == '__main__':
    input_json_path = r"C:\Users\1\Desktop\工作\dataset\中文数据集\原始数据\AI\doubao\test.json"
    output_json_path = r"C:\Users\1\Desktop\工作\dataset\中文数据集\原始数据\人类\test_data1-3\bert_distinguish_output_ai.json"
    detect_text(input_json_path, output_json_path)




