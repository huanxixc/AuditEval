import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from statistics import mean, stdev

class ResponseEvaluator:
    def __init__(self, lower_case=True, calculate_mcc=True):
        self.LOWER_CASE = lower_case
        self.CALCULATE_MCC = calculate_mcc

    def evaluate_single_pair(self, response, answer):
        gold = answer.lower() if self.LOWER_CASE else answer
        pred = response.strip().lower() if self.LOWER_CASE else response.strip()
        is_missing = 1 if not pred else 0
        pred = "missing" if is_missing else pred
        acc = 0.0 if is_missing else (1.0 if gold == pred else 0.0)
        results = {"acc": acc, "missing": is_missing, "f1": (pred, gold), "macro_f1": (pred, gold)}
        if self.CALCULATE_MCC:
            results["mcc"] = (pred, gold)
        return results, pred

    def weighted_f1(self, items):
        preds, golds = zip(*items)
        labels = list(set(golds))
        return f1_score(golds, preds, average="weighted", labels=labels)

    def macro_f1(self, items):
        preds, golds = zip(*items)
        labels = list(set(golds))
        return f1_score(golds, preds, average="macro", labels=labels)

    def matthews_corrcoef(self, items):
        preds, golds = zip(*items)
        labels = {label: i for i, label in enumerate(list(set(golds)))}
        preds = [labels.get(pred, -1) for pred in preds]
        golds = [labels.get(gold, -1) for gold in golds]
        return matthews_corrcoef(golds, preds)

    def aggregate_metrics(self, results_list):
        acc_values = [r["acc"] for r in results_list]
        missing_values = [r["missing"] for r in results_list]
        f1_pairs = [r["f1"] for r in results_list]
        macro_f1_pairs = [r["macro_f1"] for r in results_list]
        mcc_pairs = [r["mcc"] for r in results_list] if self.CALCULATE_MCC else []

        def mean_or_zero(values):
            return mean(values) if values else 0.0

        def stderr_or_zero(values):
            return stdev(values) / np.sqrt(len(values)) if len(values) > 1 else 0.0

        aggregated = {
            "acc": mean_or_zero(acc_values),
            "acc_stderr": stderr_or_zero(acc_values),
            "missing": mean_or_zero(missing_values),
            "missing_stderr": stderr_or_zero(missing_values),
            "f1": self.weighted_f1(f1_pairs) if f1_pairs else 0.0,
            "macro_f1": self.macro_f1(macro_f1_pairs) if macro_f1_pairs else 0.0,
            "mcc": self.matthews_corrcoef(mcc_pairs) if self.CALCULATE_MCC and mcc_pairs else 0.0
        }
        return aggregated

    def evaluate_from_excel(self, input_excel_path, output_excel_path, task_name="speaking_step_cognition", version=1):
        # 读取 Excel 文件
        df = pd.read_excel(input_excel_path, engine='openpyxl')

        results_list = []
        output_data = []
        for _, row in df.iterrows():
            query = str(row["query"]).strip()
            answer = str(row["answer"]).strip()
            response = str(row["response"]).strip()
            metrics, predicted = self.evaluate_single_pair(response, answer)
            results_list.append(metrics)
            output_data.append({
                "query": query,
                "answer": answer,
                "response": response,
                "predicted": predicted,
                "acc": metrics["acc"],
                "missing": metrics["missing"]
            })

        aggregated_metrics = self.aggregate_metrics(results_list)

        # 将结果保存为 Excel 文件
        # 第一部分：样本数据
        samples_df = pd.DataFrame(output_data)

        # 第二部分：聚合指标
        metrics_data = {
            "task": [task_name],
            "version": [version],
            "acc": [aggregated_metrics["acc"]],
            "acc_stderr": [aggregated_metrics["acc_stderr"]],
            "missing": [aggregated_metrics["missing"]],
            "missing_stderr": [aggregated_metrics["missing_stderr"]],
            "f1": [aggregated_metrics["f1"]],
            "macro_f1": [aggregated_metrics["macro_f1"]],
            "mcc": [aggregated_metrics["mcc"]] if self.CALCULATE_MCC else [0.0]
        }
        metrics_df = pd.DataFrame(metrics_data)

        # 写入 Excel 文件（包含两个 sheet）
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            samples_df.to_excel(writer, sheet_name='Samples', index=False)
            metrics_df.to_excel(writer, sheet_name='Aggregated Metrics', index=False)

        return aggregated_metrics

    def print_metrics(self, aggregated_metrics, task_name="speaking_step_cognition", version=1):
        print(f"Evaluation Results for {task_name} (Version {version}):")
        for metric in ["acc", "missing", "f1", "macro_f1", "mcc"]:
            if metric in aggregated_metrics:
                value = aggregated_metrics[metric]
                if metric + "_stderr" in aggregated_metrics:
                    stderr = aggregated_metrics[metric + "_stderr"]
                    print(f"{metric}: {value:.4f} ± {stderr:.4f}")
                else:
                    print(f"{metric}: {value:.4f}")


if __name__ == "__main__":

    input_excel_path = r"D:\研究生\论文材料\result\result_更新数据集\AuditWen\excel\选择任务\计算选择题.xlsx"
    output_excel_path = r"D:\研究生\论文材料\result\result_更新数据集\AuditWen\excel\选择任务\计算选择题_result.xlsx"

    evaluator = ResponseEvaluator(lower_case=True, calculate_mcc=True)
    aggregated_metrics = evaluator.evaluate_from_excel(input_excel_path, output_excel_path)
    evaluator.print_metrics(aggregated_metrics)



