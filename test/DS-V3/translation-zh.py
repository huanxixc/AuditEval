import os
import time
import json
import pandas as pd
import requests
from typing import List, Tuple
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BACKOFF_TIME = 0.05
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60

class DSChatLM:
    REQ_CHUNK_SIZE = 5  # 仅作参考，不影响串行

    def __init__(self, api_keys: List[str], truncate=False, temperature=1.0):
        self.truncate = truncate
        self.temperature = temperature
        self.api_keys = api_keys  # 多个 API 密钥列表
        self.key_index = 0  # 当前使用的密钥索引
        self.url = "https://api.deepseek.com/v1/chat/completions"  # 改为 DeepSeek 官方 API 地址
        self.model = "deepseek-chat"  # DeepSeek 官方模型名称
        self.max_gen_toks = 1024

    def _get_next_headers(self):
        """获取下一个 API 密钥的请求头"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys[self.key_index]}"
        }
        self.key_index = (self.key_index + 1) % len(self.api_keys)
        return headers

    def _send_single_request(self, payload, attempt=0):
        """发送单个请求，支持重试"""
        try:
            headers = self._get_next_headers()
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_TIME * (2 ** attempt))
                return self._send_single_request(payload, attempt + 1)
            logger.error(f"请求失败（重试 {MAX_RETRIES} 次后）: {e}")
            return ""

    def greedy_until(self, requests_list: List[Tuple[str, str]]):
        if not requests_list:
            return []

        payloads = []
        for context, until in requests_list:
            payloads.append({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的助手"},
                    {"role": "user", "content": context}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_gen_toks,
                "stream": False,
                "stop": until if until else None
            })

        # 串行处理请求
        res = []
        for payload in tqdm(payloads, desc="处理请求"):
            result = self._send_single_request(payload)
            res.append(result)
        return res

    def generate(self, prompts):
        requests_list = [(prompt, None) for prompt in prompts]
        return self.greedy_until(requests_list)

def save_to_json(file_path, data):
    """覆盖写入 JSON 文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Overwritten JSON file: {file_path}")
    except Exception as e:
        print(f"Error occurred while saving to JSON: {e}")

def save_to_excel(file_path, data):
    """覆盖写入 Excel 文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(data, columns=["query", "answer", "response"])
        df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"Overwritten Excel file: {file_path}")
    except Exception as e:
        print(f"Error occurred while saving to Excel: {e}")

def clean_response(response, query):
    if not response:
        return ""
    logging.info(f"Original response: {response}")
    query_words = set(query.split())
    response_words = response.split()
    unique_response = [word for word in response_words if word not in query_words]
    response = ' '.join(unique_response).strip()

    answer_prefixes = ["答案：", "答：", "审计案例：", "题目：", "回答："]
    for prefix in answer_prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            break
    else:
        response = response.strip()
    
    logging.info(f"Cleaned response: {response}")
    return response

def loadData(json_file_path, output_excel_path, output_json_path, batch_size, api_keys):
    try:
        print(" API client with multiple keys...")
        model = DSChatLM(api_keys=api_keys)

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Data loaded from {json_file_path}")

        all_results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            queries = [item.get("query", "") for item in batch]
            answers = [item.get("answer", "") for item in batch]

            if not any(queries) or not any(answers):
                print(f"Skipping batch {i // batch_size + 1} due to missing query or answer: {batch}")
                continue

            responses = model.generate(queries)

            batch_results = []
            for j, (query, answer, response) in enumerate(zip(queries, answers, responses)):
                cleaned_response = clean_response(response, query)
                print(f"query: '{query}'")
                print(f"Cleaned response: '{cleaned_response}'")
                batch_result = {
                    "query": query,
                    "answer": answer,
                    "response": cleaned_response
                }
                batch_results.append(batch_result)
                print(f"Processed item {i + j + 1}/{len(data)}")
            
            all_results.extend(batch_results)

        # 覆盖写入文件
        save_to_json(output_json_path, all_results)
        save_to_excel(output_excel_path, all_results)

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    # 设置多个 API 密钥（适用于硅基流动）
    api_keys = [
        "sk-3a12e7d8e743490fb14c5f16b4dd362e",
        "sk-c0614a0196fc452ebcedb0b335f4fec7",
        "sk-6855db8fde10409ab39e2f2339fa40f9",
        "sk-23a40f815d714f11a3dd97670aef8869"
    ]

    # 文件路径配置
    # 中译英
    # filename1 = "/root/autodl-tmp/PIXIU/data/文本翻译/中译英/标题翻译/test.json"
    # output_excel_path = "/root/autodl-tmp/ds-r1测评/data/result/excel/文本翻译/中译英/tr_title_zh_translation.xlsx"
    # output_json_path = "/root/autodl-tmp/ds-r1测评/data/result/json/文本翻译/中译英/tr_title_zh_translation.json"  ok 晚

    # filename1 = "/root/autodl-tmp/PIXIU/data/文本翻译/中译英/法规翻译/test.json"
    # output_excel_path = "/root/autodl-tmp/ds-r1测评/data/result/excel/文本翻译/中译英/tr_law_zh_translation.xlsx"
    # output_json_path = "/root/autodl-tmp/ds-r1测评/data/result/json/文本翻译/中译英/tr_law_zh_translation.json"

    # filename1 = "/root/autodl-tmp/PIXIU/data/文本翻译/中译英/关键词翻译/test.json"
    # output_excel_path = "/root/autodl-tmp/ds-r1测评/data/result/excel/文本翻译/中译英/tr_keywords_zh_translation.xlsx"
    # output_json_path = "/root/autodl-tmp/ds-r1测评/data/result/json/文本翻译/中译英/tr_keywords_zh_translation.json"  ok 完

    # filename1 = "/root/autodl-tmp/PIXIU/data/文本翻译/中译英/摘要翻译/test.json"
    # output_excel_path = "/root/autodl-tmp/ds-v3测评/result/excel/文本翻译/中译英/tr_abstract_zh_translation.xlsx"
    # output_json_path = "/root/autodl-tmp/ds-v3测评/result/json/文本翻译/中译英/tr_abstract_zh_translation.json"

    # filename1 = "/root/autodl-tmp/PIXIU/data/文本翻译/英译中/摘要翻译/test.json"
    # output_excel_path = "/root/autodl-tmp/ds-v3测评/result/excel/文本翻译/英译中/tr_abstract_en_translation.xlsx"
    # output_json_path = "/root/autodl-tmp/ds-v3测评/result/json/文本翻译/英译中/tr_abstract_en_translation.json"

    filename1 = "/root/autodl-tmp/PIXIU/data/文本翻译/英译中/法规翻译/test.json"
    output_excel_path = "/root/autodl-tmp/ds-v3测评/result/excel/文本翻译/英译中/tr_law_en_translation.xlsx"
    output_json_path = "/root/autodl-tmp/ds-v3测评/result/json/文本翻译/英译中/tr_law_en_translation.json"

    batch_size = 32

    loadData(filename1, output_excel_path, output_json_path, batch_size, api_keys)