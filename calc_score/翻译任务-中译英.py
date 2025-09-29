# import os
# import json
# import sys
# import jieba
# import pandas as pd  # 添加 pandas 用于读取 Excel
# from bert_score import score
# from rouge_chinese import Rouge
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from statistics import mean
#
# # 环境配置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_OFFLINE'] = '1'
#
# class QAwithString():
#     def rougeChinese(self, items):
#         # 中文分词（样本显示 answer 和 response 是中文）
#         jieba.load_userdict(r"D:\研究生\论文材料\PIXIU\审计分词表.txt")
#         hyps, refs = map(list, zip(*[[' '.join(jieba.cut(d[0])), ' '.join(jieba.cut(d[1]))] for d in items]))
#         filter_hyps = []
#         filter_refs = []
#         for hyp, ref in zip(hyps, refs):
#             hyp_str = hyp.strip()
#             ref_str = ref.strip()
#             if hyp_str and ref_str:
#                 filter_hyps.append(hyp_str)
#                 filter_refs.append(ref_str)
#
#         rouge = Rouge()
#         scores = rouge.get_scores(filter_hyps, filter_refs, avg=True, ignore_empty=True)
#         return scores
#
#     def bert_score(self, items):
#         golds, preds = zip(*items)
#         P, R, F1 = score(golds, preds, lang="zh", model_type="bert-base-chinese", verbose=True)  # 中文模型
#         output_dict = {
#             "precision": P.tolist(),
#             "recall": R.tolist(),
#             "f1": F1.tolist(),
#         }
#         return sum(output_dict["f1"]) / len(output_dict["f1"])
#
#     def bleu_score(self, items):
#         jieba.load_userdict(r"D:\研究生\论文材料\PIXIU\审计分词表.txt")
#         smoothing_function = SmoothingFunction().method1
#         scores = []
#         for ref, hyp in items:
#             ref_tokens = list(jieba.cut(ref.lower()))  # 中文分词
#             hyp_tokens = list(jieba.cut(hyp.lower()))  # 中文分词
#             score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing_function)
#             scores.append(score)
#         return sum(scores) / len(scores) if scores else 0
#
#     def accuracy(self, items):
#         acc_scores = []
#         for ref, hyp in items:
#             ref = ref.lower().strip()
#             hyp = hyp.lower().strip()
#             acc = 1.0 if ref == hyp else 0.0
#             acc_scores.append(acc)
#         return mean(acc_scores) if acc_scores else 0
#
# def compute_method(excel_file_path):
#     qa_task = QAwithString()
#
#     # 读取 Excel 文件
#     df = pd.read_excel(excel_file_path, engine='openpyxl')
#
#     items = []
#     for _, row in df.iterrows():
#         answer = str(row['answer']).strip()  # 对应 Excel 中的 answer 列
#         response = str(row['response']).strip()  # 对应 Excel 中的 response 列
#         if answer and response:
#             items.append((answer, response))
#
#     bert_score_result = qa_task.bert_score(items)
#     rouge_result = qa_task.rougeChinese(items)
#     bleu_result = qa_task.bleu_score(items)
#     acc_result = qa_task.accuracy(items)
#     return bert_score_result, rouge_result, bleu_result, acc_result
#
# if __name__ == '__main__':
#     # excel_file_path = r"D:\研究生\论文材料\result\result_更新数据集\deepseek-r1\文本翻译\中译英\tr_title_zh_translation.xlsx"
#
#     # excel_file_path = r"D:\研究生\论文材料\result\result_更新数据集\deepseek-r1\文本翻译\中译英\tr_title_zh_translation.xlsx"
#
#     # excel_file_path = r"D:\研究生\论文材料\result\result_更新数据集\deepseek-r1\文本翻译\中译英\tr_title_zh_translation.xlsx"
#     #
#     excel_file_path = r"D:\研究生\论文材料\result\result_更新数据集\gpt4\翻译任务\tr_abstract_zh_translation_gpt4\tr_abstract_zh_translation.xlsx"
#
#
#     bert_score, rouge_chinese, bleu_score, acc = compute_method(excel_file_path)
#
#     print(excel_file_path)
#     print("ROUGE-1: {:.4f}".format(rouge_chinese["rouge-1"]['f']))
#     print("ROUGE-2: {:.4f}".format(rouge_chinese["rouge-2"]['f']))
#     print("ROUGE-L: {:.4f}".format(rouge_chinese["rouge-l"]['f']))
#     print("BERT Score (F1): {:.4f}".format(bert_score))
#     print("BLEU Score: {:.4f}".format(bleu_score))
#     print("Accuracy: {:.4f}".format(acc))
#     print("********************************************************")


import os
import pandas as pd
from bert_score import score
from rouge import Rouge  # 英文 Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from statistics import mean

# 环境配置（可选）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'

class QAwithString():
    def rouge_english(self, items):
        hyps, refs = zip(*items)
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        return scores

    def bert_score(self, items):
        golds, preds = zip(*items)
        P, R, F1 = score(golds, preds, lang="en", model_type="bert-base-uncased", verbose=True)
        output_dict = {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
        }
        return sum(output_dict["f1"]) / len(output_dict["f1"])

    def bleu_score(self, items):
        smoothing_function = SmoothingFunction().method1
        scores = []
        for ref, hyp in items:
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()
            score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing_function)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0

    def accuracy(self, items):
        acc_scores = []
        for ref, hyp in items:
            ref = ref.lower().strip()
            hyp = hyp.lower().strip()
            acc = 1.0 if ref == hyp else 0.0
            acc_scores.append(acc)
        return mean(acc_scores) if acc_scores else 0

def compute_method(excel_file_path):
    qa_task = QAwithString()

    # 读取 Excel 文件
    df = pd.read_excel(excel_file_path, engine='openpyxl')

    items = []
    for _, row in df.iterrows():
        answer = str(row['answer']).strip()
        response = str(row['response']).strip()
        if answer and response:
            items.append((answer, response))

    bert_score_result = qa_task.bert_score(items)
    rouge_result = qa_task.rouge_english(items)
    bleu_result = qa_task.bleu_score(items)
    acc_result = qa_task.accuracy(items)
    return bert_score_result, rouge_result, bleu_result, acc_result

if __name__ == '__main__':
    # 评测qwen3-8b
    excel_file_path = r"D:\研究生\论文材料\result\result_更新数据集\AuditWen\excel\文本翻译\中译英\摘要翻译.xlsx"
    bert_score, rouge_english, bleu_score, acc = compute_method(excel_file_path)

    print(excel_file_path)
    print("ROUGE-1: {:.4f}".format(rouge_english["rouge-1"]['f']))
    print("ROUGE-2: {:.4f}".format(rouge_english["rouge-2"]['f']))
    print("ROUGE-L: {:.4f}".format(rouge_english["rouge-l"]['f']))
    print("BERT Score (F1): {:.4f}".format(bert_score))
    print("BLEU Score: {:.4f}".format(bleu_score))
    print("Accuracy: {:.4f}".format(acc))
    print("********************************************************")
