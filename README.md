# AuditEval
AuditEval: A Benchmark for Auditing Tasks in LLMs
## Introduction

AuditEval is a comprehensive benchmark specifically designed to evaluate the capabilities of large language models (LLMs) in the auditing domain. The purpose of this project is to provide a multidimensional framework for evaluating LLMs across a variety of auditing tasks. The framework focuses on key competencies such as auditing professional knowledge, practical application abilities, and academic expression skills. It also addresses gaps in current LLM evaluation benchmarks by offering a structured task system and fine-grained performance assessment.

This evaluation framework is essential for deploying LLMs in intelligent auditing applications, ensuring that they can effectively handle domain-specific challenges and provide high-quality outputs for real-world auditing tasks.
## Task

The **AuditEval** framework is built to systematically assess LLMs on various auditing-related tasks. It is based on a hierarchical system of tasks divided into three primary evaluation dimensions:

1. **Professional Knowledge** – Assesses the model's ability to handle domain-specific auditing knowledge, including concept identification, question answering, and regulation-based reasoning.
2. **Practical Application** – Focuses on the model's real-world applicability, such as entity recognition, text generation, and issue qualification, to ensure the model can handle complex auditing scenarios.
3. **Academic Expression** – Measures the model's capability to engage in academic-style tasks, including move recognition and bilingual translation, critical for academic and professional auditing report generation.

The framework includes 8 main tasks and 32 sub-tasks, spanning across multiple knowledge areas such as accounting, law, and taxation. Each task is associated with specific metrics (e.g., accuracy, ROUGE, BLEU, etc.), and performance is evaluated based on both **task difficulty** and **answer openness**.

## Task table
|               **Task Domain**                 |                 **Data Domain**                       |                      **Metric Domain**                                 |
| **Task<br>Dimension**  | **Task** | **SubTask**  |   **Dataset**<br>**Size**   |     **Source of Data**         | **Quantitative<br>Metric**   |   **Task<br>Difficulty**  |**Answer<br>Openness**|
|---------------------|----------|--------------|----------------------|--------------------------------|-------------- ------------|------------------------|-------------------|


### Model Evaluation

In this project, we evaluate the performance of various LLMs across different auditing tasks. Models are assessed using the **AuditEval** benchmark, which includes tasks that measure knowledge, practical application, and academic expression. The evaluation helps in understanding how well different models handle auditing-specific challenges and how their performance varies across task types and difficulty levels.

### Models for Evaluation

You can download the models used in the evaluation from the following links:

- **Qwen2.5-7B-Instruct**: [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Meta-Llama-3.1-8B-Instruct**: [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **ChatGLM3-6B**: [https://huggingface.co/zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b)
- **Qwen3-8B**: [https://huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **AuditWen** (Fine-tuned on auditing data): [https://huggingface.co/HooRin/AuditWen](https://huggingface.co/HooRin/AuditWen)
- **DeepSeek-V3**: [https://huggingface.co/deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- **DeepSeek-R1**: [https://huggingface.co/deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- **GPT-4**: Closed-Source · API Access Required

### Evaluation Results

## Quick Start

## License

## Citation

## Contact
