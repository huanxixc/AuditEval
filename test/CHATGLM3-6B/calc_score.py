import json
import os
import logging
import pandas as pd
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
import jieba
from rouge_chinese import Rouge
import bert_score
from pytablewriter import MarkdownTableWriter

# 设置环境变量以优化 CUDA 内存管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 封装写入JSON文件的功能
def write_to_json(file_path, data):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"JSON file written to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while writing to JSON: {e}")

# 封装写入Excel文件的功能
def write_to_excel(file_path, data):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)
        logging.info(f"Excel file written to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while writing to Excel: {e}")

# 封装清理响应的功能
def clean_response(response, query):
    logging.info(f"Original response: {response}")
    # 移除特定的前缀部分
    prefix = "[gMASK]sop user: "
    if response.startswith(prefix):
        response = response[len(prefix):]

    # 先移除响应中与查询重复的部分
    query_words = set(query.split())  # 将查询分割成单词并去重
    response_words = response.split()  # 响应分割成单词
    unique_response = [word for word in response_words if word not in query_words]  # 移除与查询重复的单词
    response = ' '.join(unique_response).strip()  # 重新组合成字符串

    # 再检查是否包含“答案：”或“答：”并处理
    answer_prefixes = ["答案：", "答：", "审计案例：", "题目：", "回答："]  # 定义可能的前缀列表
    for prefix in answer_prefixes:  # 遍历前缀列表
        if response.startswith(prefix):  # 如果响应以某个前缀开头
            response = response[len(prefix):].strip()  # 移除前缀并去空格
            break  # 处理完一个前缀后就退出循环
    else:  # 如果没有找到任何前缀
        response = response.strip()  # 只去除首尾空格
    
    cleaned_response = response
    logging.info(f"Cleaned response: {cleaned_response}")
    return cleaned_response

# 主类：封装通用逻辑
class ChatGLMTask:
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size
        self.load_model()

    def load_model(self):
        logging.info(f"Loading model and tokenizer from {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # chatglm-6b 推荐使用 float16
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side="left",
            trust_remote_code=True  # 确保加载自定义代码
        )
        self.tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"

    def generate_responses(self, json_file_path, output_excel_path, output_json_path):
        try:
            # 从 JSON 文件中加载数据
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info(f"Data loaded from {json_file_path}")

            # 按批次处理数据
            results = []  # 用于存储所有结果
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                queries = [item.get("query") for item in batch]
                answers = [item.get("answer") for item in batch]

                # 检查是否有缺失的查询或答案
                if None in queries or None in answers:
                    logging.warning(f"Skipping items due to missing query or answer: {batch}")
                    continue

                # 构建输入
                messages = [{"role": "user", "content": q} for q in queries]
                texts = [self.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]
                model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

                # 生成响应
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024
                )
                responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # 处理每条生成的响应
                for j, response in enumerate(responses):
                    cleaned_response = clean_response(response, queries[j])
                    results.append({
                        "query": queries[j],
                        "answer": answers[j],
                        "response": cleaned_response
                    })
                    logging.info(f"Processed item {i + j + 1}/{len(data)}")

            # 写入 JSON 文件
            write_to_json(output_json_path, results)

            # 写入 Excel 文件
            write_to_excel(output_excel_path, results)

        except Exception as e:
            logging.error(f"Error occurred: {e}")

    def evaluate(self, dataset_path, task_name):
        qa_task = ChatGLMTest(dataset_path=dataset_path)
        items = qa_task.dataset  # 直接使用加载的数据集

        # 计算指标
        rouge_chinese = qa_task.rougeChinese(items)
        bert_score_f1 = qa_task.bert_score_f1(items)

        # 准备结果字典
        results = {
            "rouge-1": rouge_chinese["rouge-1"],
            "rouge-2": rouge_chinese["rouge-2"],
            "rouge-l": rouge_chinese["rouge-l"],
            "bert_score_f1": bert_score_f1
        }

        # 生成结果表格
        results_table = qa_task.make_table(results, task_name)
        print(results_table)

# 封装计算指标的功能
class ChatGLMTest:
    DATASET_NAME = None
    EVAL_LAST_TURN = True

    def __init__(self, dataset_path=None):
        self.DATASET_NAME = dataset_path
        if dataset_path is None:
            logging.info("No dataset path provided. Skipping dataset loading.")
            self.dataset = []
        else:
            logging.info(f"Loading dataset: {dataset_path}")
            if not os.path.exists(dataset_path):
                logging.error(f"Dataset file does not exist: {dataset_path}")
                raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
            try:
                with open(dataset_path, 'r', encoding='utf-8') as file:
                    self.dataset = json.load(file)
            except Exception as e:
                logging.error(f"Failed to load dataset {dataset_path}: {e}")
                raise

    def is_whitespace_string(self, s):
        return s.isspace()

    def rougeChinese(self, items):
        if not items:
            logging.warning("No items provided for ROUGE-Chinese evaluation.")
            return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}

        # 加载自定义分词表
        try:
            jieba.load_userdict("/root/autodl-tmp/PIXIU/models/审计分词表.txt")
        except Exception as e:
            logging.warning(f"加载自定义词典失败：{e}")

        hyps = []
        refs = []
        for item in items:
            if isinstance(item, dict) and "answer" in item and "response" in item:
                hyps.append(' '.join(jieba.cut(item["answer"])))
                refs.append(' '.join(jieba.cut(item["response"])))
            else:
                logging.warning("Item does not contain 'answer' and 'response' keys.")
        filter_hyps = [hyp for hyp in hyps if hyp.strip()]
        filter_refs = [ref for ref in refs if ref.strip()]

        rouge = Rouge()
        scores = rouge.get_scores(filter_hyps, filter_refs, avg=True, ignore_empty=True)
        return scores

    def bert_score_f1(self, items):
        golds = []
        preds = []
        for item in items:
            if isinstance(item, dict) and "answer" in item and "response" in item:
                golds.append(item["answer"])
                preds.append(item["response"])
            else:
                logging.warning("Item does not contain 'answer' and 'response' keys.")
        try:
            P, R, F1 = bert_score.score(
                golds, preds, lang="zh", model_type="bert-base-chinese", verbose=True
            )
        except Exception as e:
            logging.error(f"BERTScore 计算失败：{e}")
            return 0.0
        return sum(F1.tolist()) / len(F1.tolist())

    @staticmethod
    def make_table(results, task_name):
        """Generate table of results."""
        md_writer = MarkdownTableWriter()
        md_writer.headers = ["Task", "Metric", "Value", "Stderr"]

        values = [
            [task_name, "rouge1", f"{results['rouge-1']['f']:.4f}", ""],
            ["", "rouge2", f"{results['rouge-2']['f']:.4f}", ""],
            ["", "rougeL", f"{results['rouge-l']['f']:.4f}", ""],
            ["", "bert_score_f1", f"{results['bert_score_f1']:.4f}", ""],
        ]

        md_writer.value_matrix = values

        return md_writer.dumps()

# 子类：具体任务
##自动问答
class Audit_Standard(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_Standard1"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/自动问答/审计准则指南/audit_standard1.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/自动问答/Audit_Standard1.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/自动问答/Audit_Standard1.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class Audit_law(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_law"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/自动问答/审计法规问答/test.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/自动问答/Audit_law.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/自动问答/Audit_law.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class Audit_conception(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_conception"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/自动问答/审计概念问答/QA_concept1.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/自动问答/Audit_conception1.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/自动问答/Audit_conception1.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class Audit_Target(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_Target"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/自动问答/审计目标问答/audit_target.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/自动问答/audit_target.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/自动问答/audit_target.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class Audit_content(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_content"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/自动问答/审计内容问答/audit_content.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/自动问答/Audit_content.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/自动问答/Audit_content.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)


##文本生成任务
class Audit_problem_summary(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_problem_summary"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/文本生成/审计问题总结/audit problem summary.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/文本生成/Audit_problem_summary.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/文本生成/Audit_problem_summary.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class Audit_material_analyse(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_material_analyse"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/文本生成/审计材料分析/audit_material_analyse.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/文本生成/Audit_material_analyse.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/文本生成/Audit_material_analyse.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class Audit_case_generation(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_case_generation"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/文本生成/审计案例撰写/test.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/文本生成/Audit_case_generation.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/文本生成/Audit_case_generation.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

##问题定性
class Audit_matters(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_matters"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/问题定性/审计事项/test.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/问题定性/Audit_matters.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/问题定性/Audit_matters.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class Audit_manifestations(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Audit_manifestations"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/问题定性/问题表现形式/manifestations_test.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/问题定性/Audit_manifestations.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/问题定性/Audit_manifestations.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class simple_Qualitative_basis(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "simple_Qualitative_basis"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/问题定性/简单_问题定性依据/simple_qualitative_basis.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/问题定性/simple_Qualitative_basis.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/问题定性/simple_Qualitative_basis.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class Qualitative_basis(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "Qualitative_basis"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/问题定性/问题定性依据/Qualitative_basis.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/问题定性/Qualitative_basis.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/问题定性/Qualitative_basis.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class simple_penalty_basis(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "simple_penalty_basis"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/问题定性/简单_问题处罚依据/simple_penalty_test.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/问题定性/simple_penalty_basis.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/问题定性/simple_penalty_basis.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

class penalty_basis(ChatGLMTask):
    def __init__(self, model_name, batch_size):
        super().__init__(model_name, batch_size)
        self.task_name = "penalty_basis"
        self.json_file_path = "/root/autodl-tmp/test_models/new_data/问题定性/问题处罚依据/penalty_test.json"
        self.output_excel_path = "/root/autodl-tmp/test_models/result/excel/问题定性/penalty_basis.xlsx"
        self.output_json_path = "/root/autodl-tmp/test_models/result/json/问题定性/penalty_basis.json"

    def run(self):
        self.generate_responses(self.json_file_path, self.output_excel_path, self.output_json_path)
        self.evaluate(self.output_json_path, self.task_name)

# 主函数
def main():
    model_name = "/root/autodl-tmp/PIXIU/models/ZhipuAI/chatglm3-6b"  # 模型名称
    batch_size = 32  # 调整批量大小以减少 CUDA 内存占用

    # 创建并运行具体任务
    audit_task = Audit_Standard(model_name, batch_size)
    audit_task.run()

if __name__ == '__main__':
    main()