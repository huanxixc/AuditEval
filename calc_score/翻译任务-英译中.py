import os
import sys
import jieba
import pandas as pd  # 添加 pandas 用于读写 Excel
from bert_score import score
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from statistics import mean

# 环境配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'

class QAwithString():
    def rougeChinese(self, items):
        jieba.load_userdict(r"D:\研究生\论文材料\PIXIU\审计分词表.txt")
        hyps, refs = map(list, zip(*[[' '.join(jieba.cut(d[0])), ' '.join(jieba.cut(d[1]))] for d in items]))

        filter_hyps = []
        filter_refs = []
        for i in range(len(hyps)):
            hyp = hyps[i]
            ref = refs[i]
            if hyp.strip() and ref.strip():
                filter_hyps.append(hyp)
                filter_refs.append(ref)

        rouge = Rouge()
        scores = rouge.get_scores(filter_hyps, filter_refs, avg=True, ignore_empty=True)
        return scores

    def bert_score(self, items):
        golds, preds = zip(*items)
        P, R, F1 = score(golds, preds, lang="zh", model_type="bert-base-chinese", verbose=True)
        output_dict = {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
        }
        return sum(output_dict["f1"]) / len(output_dict["f1"])

    def bleu_score(self, items):
        jieba.load_userdict(r"D:\研究生\论文材料\PIXIU\审计分词表.txt")
        smoothing_function = SmoothingFunction().method1
        scores = []
        for ref, hyp in items:
            ref_tokens = jieba.cut(ref.lower())
            hyp_tokens = jieba.cut(hyp.lower())
            score = sentence_bleu([list(ref_tokens)], list(hyp_tokens), smoothing_function=smoothing_function)
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

def compute_method(excel_file_path, output_excel_path):
    qa_task = QAwithString()

    # 加载 Excel 数据
    df = pd.read_excel(excel_file_path, engine='openpyxl')

    items = []
    for _, row in df.iterrows():
        answer = str(row["answer"]).strip()
        response = str(row["response"]).strip()
        if answer and response:
            items.append((answer, response))

    bert_score_result = qa_task.bert_score(items)
    rouge_result = qa_task.rougeChinese(items)
    bleu_result = qa_task.bleu_score(items)
    acc_result = qa_task.accuracy(items)

    # 将结果保存为 Excel 文件
    metrics_data = {
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERT Score (F1)", "BLEU Score", "Accuracy"],
        "Value": [
            rouge_result["rouge-1"]['f'],
            rouge_result["rouge-2"]['f'],
            rouge_result["rouge-l"]['f'],
            bert_score_result,
            bleu_result,
            acc_result
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_excel(output_excel_path, index=False, engine='openpyxl')

    return bert_score_result, rouge_result, bleu_result, acc_result

if __name__ == '__main__':
    # 评测qwen3-8b
    excel_file_path = r"D:\研究生\论文材料\result\result_更新数据集\AuditWen\excel\文本翻译\英译中\法规翻译.xlsx"
    output_excel_path = r"D:\研究生\论文材料\result\result_更新数据集\AuditWen\excel\文本翻译\英译中\法规翻译_result.xlsx"

    bert_score, rouge_chinese, bleu_score, acc = compute_method(excel_file_path, output_excel_path)

    print(excel_file_path)
    print("ROUGE-1: {:.4f}".format(rouge_chinese["rouge-1"]['f']))
    print("ROUGE-2: {:.4f}".format(rouge_chinese["rouge-2"]['f']))
    print("ROUGE-L: {:.4f}".format(rouge_chinese["rouge-l"]['f']))
    print("BERT Score (F1): {:.4f}".format(bert_score))
    print("BLEU Score: {:.4f}".format(bleu_score))
    print("Accuracy: {:.4f}".format(acc))
    print("********************************************************")