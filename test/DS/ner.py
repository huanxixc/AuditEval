import os
import time
import json
import pandas as pd
import requests
from typing import List
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BACKOFF_TIME = 0.05
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60

class DSChatLM:
    def __init__(self, api_keys: List[str], temperature=1.0):
        self.temperature = temperature
        self.api_keys = api_keys
        self.key_index = 0
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.max_gen_toks = 1024

    def _get_next_headers(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys[self.key_index]}"
        }
        self.key_index = (self.key_index + 1) % len(self.api_keys)
        return headers

    def _send_single_request(self, payload, attempt=0):
        try:
            headers = self._get_next_headers()
            response = requests.post(self.url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_TIME * (2 ** attempt))
                return self._send_single_request(payload, attempt + 1)
            logger.error(f"请求失败: {e}")
            return ""

    def greedy_until(self, prompts: List[str]):
        payloads = [
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的命名实体识别助手。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_gen_toks,
                "stream": False
            }
            for prompt in prompts
        ]

        responses = []
        for payload in tqdm(payloads, desc="处理NER请求"):
            response = self._send_single_request(payload)
            responses.append(response)

        return responses

    def generate(self, prompts):
        return self.greedy_until(prompts)

# 加载 few-shot 示例
def load_fewshot_examples(train_file_path, num_shots=5):
    with open(train_file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset[:num_shots]

# 构造 NER 输入
def construct_ner_input(doc, examples):
    input_text = "以下是命名实体识别任务的示例，请参考这些示例完成任务。\n\n示例：\n"
    for ex in examples:
        input_text += f" Text: {ex['text']} Answer: {ex['answer']}"
    input_text += f" Text: {doc['text']} Answer:"
    return input_text

def evaluate_and_save(train_file, test_file, task_name, api_keys, output_dir, batch_size):
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing DeepSeek API client with multiple keys...")
    model = DSChatLM(api_keys=api_keys)
    examples = load_fewshot_examples(train_file)

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        inputs = [construct_ner_input(doc, examples) for doc in batch]
        responses = model.generate(inputs)

        for idx, (doc, prompt, response) in enumerate(zip(batch, inputs, responses)):
            results.append({
                "doc_id": i + idx,
                "prompt_0": prompt,
                "response": response.strip(),
                "answer": doc["answer"]
            })

    json_path = os.path.join(output_dir, f"{task_name}_results.json")
    excel_path = os.path.join(output_dir, f"{task_name}_results.xlsx")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Overwritten JSON file: {json_path}")

    df = pd.DataFrame(results, columns=["doc_id", "prompt_0", "response", "answer"])
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Overwritten Excel file: {excel_path}")

if __name__ == "__main__":
    api_keys = [
        "sk-82a5196585df41839afe7ba491445b35",
        "sk-37ede6ed162a4778b3bfec0da8d68895",
        "sk-efab8262eb6c43eeac8174399ada4b8f"
    ]

    # train_file = "/root/autodl-tmp/PIXIU/data/NER/NER_3/train.json"
    # test_file = "/root/autodl-tmp/PIXIU/data/NER/NER_3/test.json"
    # output_dir = "/root/autodl-tmp/ds-v3测评/result/ner"
    # task_name = "NER_3"

    # train_file = "/root/autodl-tmp/PIXIU/data/NER/auditner_7/train.json"
    # test_file = "/root/autodl-tmp/PIXIU/data/NER/auditner_7/test.json"
    # output_dir = "/root/autodl-tmp/ds-r1测评/data/result/ner"
    # task_name = "NER_7"

    train_file = "/root/autodl-tmp/PIXIU/data/NER/auditner_20/train.json"
    test_file = "/root/autodl-tmp/PIXIU/data/NER/auditner_20/test.json"
    output_dir = "/root/autodl-tmp/ds-v3测评/result/ner"
    task_name = "NER_20"
    
    batch_size = 32

    evaluate_and_save(train_file, test_file, task_name, api_keys, output_dir, batch_size)
