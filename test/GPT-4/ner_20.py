import os
import asyncio
import json
import pandas as pd
import httpx
from typing import List
from tqdm.asyncio import tqdm_asyncio  # 修改为异步兼容的tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BACKOFF_TIME = 0.05
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60

class GPT4ChatLM:
    def __init__(self, api_keys: List[str], temperature=1.0):
        self.temperature = temperature
        self.api_keys = api_keys
        self.key_index = 0
        self.url = "https://api.132999.xyz/v1/chat/completions"
        self.model = "gpt-4"
        self.max_gen_toks = 1024

    def _get_next_headers(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys[self.key_index]}"
        }
        self.key_index = (self.key_index + 1) % len(self.api_keys)
        return headers

    async def _send_single_request(self, client, payload, pbar, attempt=0):  # 新增pbar参数
        try:
            headers = self._get_next_headers()
            response = await client.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BACKOFF_TIME * (2 ** attempt))
                return await self._send_single_request(client, payload, pbar, attempt + 1)
            logger.error(f"请求失败: {e}")
            return ""
        finally:  # 确保进度条更新
            pbar.update(1)

    async def generate(self, prompts: List[str]):
        payloads = [
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的命名实体识别助手。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_gen_toks,
            }
            for prompt in prompts
        ]

        # 添加请求级进度条
        with tqdm_asyncio(total=len(payloads), desc="API请求进度", unit="req") as pbar:
            async with httpx.AsyncClient() as client:
                tasks = [self._send_single_request(client, payload, pbar) for payload in payloads]
                responses = await asyncio.gather(*tasks)
                return responses

def load_fewshot_examples(train_file_path, num_shots=5):
    with open(train_file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset[:num_shots]

def construct_ner_input(doc, examples):
    input_text = "以下是命名实体识别任务的示例，请参考这些示例完成任务。\n\n示例：\n"
    for ex in examples:
        input_text += f" Text: {ex['text']} Answer: {ex['answer']}"
    input_text += f" Text: {doc['text']} Answer:"
    return input_text

async def evaluate_and_save(train_file, test_file, task_name, api_keys, output_dir, batch_size):
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing GPT-4 API client with multiple keys...")
    model = GPT4ChatLM(api_keys=api_keys)
    examples = load_fewshot_examples(train_file)

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    
    # 添加总进度条
    with tqdm_asyncio(total=len(test_data), desc="总进度", unit="doc") as main_pbar:
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            inputs = [construct_ner_input(doc, examples) for doc in batch]
            
            responses = await model.generate(inputs)

            for idx, (doc, prompt, response) in enumerate(zip(batch, inputs, responses)):
                results.append({
                    "doc_id": i + idx,
                    "prompt_0": prompt,
                    "response": response.strip(),
                    "answer": doc["answer"]
                })
            
            # 更新总进度
            main_pbar.update(len(batch))

    json_path = os.path.join(output_dir, f"{task_name}_results.json")
    excel_path = os.path.join(output_dir, f"{task_name}_results.xlsx")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON file: {json_path}")

    df = pd.DataFrame(results, columns=["doc_id", "prompt_0", "response", "answer"])
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Saved Excel file: {excel_path}")

if __name__ == "__main__":
    api_keys = [
        "sk-K2cpsmKoIjVaCaxVkdHR9CLyazkHc9nqpp2HTbXjoVLJzwBV"
    ]

    # 配置路径参数（根据实际情况修改）
    train_file = "/root/autodl-tmp/PIXIU/data/NER/auditner_20/train.json"
    test_file = "/root/autodl-tmp/PIXIU/data/NER/auditner_20/test.json"
    output_dir = "/root/autodl-tmp/test_model/result/ner/ner_20"
    task_name = "NER_20"
    batch_size = 32

    asyncio.run(evaluate_and_save(
        train_file, test_file, task_name, api_keys, 
        output_dir, batch_size
    ))
