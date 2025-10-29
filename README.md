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
<table>
    <tr>
        <td colspan="3">Task Domain</td> 
        <td colspan="2">Data Domain</td> 
        <td colspan="3">Metric Domain</td> 
   </tr>
    <tr>
  		  <td>Task Dimension</td> 
        <td>Task</td>    
        <td>SubTask</td> 
        <td>Dataset Size</td> 
        <td>Source of Data</td>
        <td>Quantitative Metric</td>
        <td>Task Difficulty</td>
        <td>Answer Openness</td>
    </tr>
    <tr>
        <td>跨三列合并行</td> 
        <td>跨三列合并行</td>
        <td>跨三列合并行</td>
        <td>跨三列合并行</td>
        <td>跨三列合并行</td>
        <td>跨三列合并行</td>
        <td>跨三列合并行</td>
        <td>跨三列合并行</td>
    </tr>
</table>


### AuditWen Benchmark 
<div align="center">
![Fig 2: AuditWen Benchmark](https://github.com/huanxixc/AuditEval/raw/main/figures/Fig.2.png)
</div>

## Model Evaluation

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

## Evaluation Results
### Professional Knowledge Tasks

In the **Professional Knowledge** dimension, models were evaluated on tasks like multiple-choice questions, true/false questions, and automatic question answering (QA). The results show that larger models generally perform better, but **AuditWen**, a domain-specific fine-tuned model, showed strong performance across multiple subtasks. **DeepSeek-R1** outperformed other models in numerical calculation tasks, achieving the highest accuracy.

- **AuditWen** excelled in answering auditing standards and concepts, demonstrating the effectiveness of domain-specific fine-tuning.
- **DeepSeek-R1** showed strong performance in reasoning tasks due to its large-scale parameters and sophisticated training.

### Practical Application Tasks

For the **Practical Application** dimension, models were evaluated on tasks such as entity recognition, audit phrase classification, and issue qualification. Here, **AuditWen** performed exceptionally well, thanks to its fine-tuning on auditing-specific datasets. Models like **Qwen3-8B** also showed strong results in classification tasks.

- **AuditWen** dominated entity recognition tasks, performing well in recognizing and classifying entities relevant to auditing.
- **DeepSeek-R1** performed well in audit issue qualification, leveraging its large model size and refined training.

### Academic Expression Tasks

In the **Academic Expression** dimension, models were evaluated on tasks like move recognition and bilingual translation. **DeepSeek-V3** performed the best in move recognition tasks, accurately classifying academic moves in audit-related texts. **Qwen3-8B** performed well in bilingual translation tasks, particularly in translating auditing terms.

- **DeepSeek-V3** demonstrated superior accuracy in move recognition, outperforming all other models.
- **Qwen3-8B** and **DeepSeek-V3** performed well in bilingual translation tasks, showing high alignment with reference translations in both Chinese-to-English and English-to-Chinese tasks.

### Final Model Performance

The overall model performance was aggregated across all tasks, considering both task difficulty and answer openness. **DeepSeek-R1** ranked the highest overall, especially in more complex, reasoning-based tasks, followed by **AuditWen**, which showed strong results in auditing-specific tasks due to its domain-focused fine-tuning.

- **DeepSeek-R1** performed well in high-difficulty tasks, benefiting from its large parameter size and complex reasoning abilities.
- **AuditWen** was particularly effective in practical auditing applications, such as entity recognition and issue qualification, where domain-specific knowledge was critical.
- **Qwen3-8B** demonstrated solid performance across all tasks, with particular strengths in bilingual translation and professional knowledge tasks.

## Quick Start

## License

## Citation

## Contact
