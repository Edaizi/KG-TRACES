# ✨ KG-TRACES: Unleashing Explainable Reasoning in LLMs with Knowledge Graphs ✨

[![arXiv](https://img.shields.io/badge/arXiv-2506.00783-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2506.00783)
[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge)](https://www.python.org)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.60+-ee4c2c.svg?style=for-the-badge)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-KG--TRACES-blue?style=for-the-badge)](https://huggingface.co/Edaizi)


Welcome to the official repository for **KG-TRACES**! 🚀 We're enhancing Large Language Models to reason with **explainable**, **accuracy**, and **traceability** by leveraging the power of Knowledge Graphs.


Rong Wu, Pinlong Cai, Jianbiao Mei, Licheng Wen, Tao Hu, Xuemeng Yang, Daocheng Fu, Botian Shi

*Zhejiang University, Shanghai Artificial Intelligence Laboratory, University of Science and Technology of China, Fudan University*

---

## 🎯 The Challenge: LLMs Lost in Thought?

Vanilla LLMs are amazing, but when it comes to complex, multi-hop reasoning, they can sometimes...
*   🤯 Hallucinate facts
*   ❓ Provide answers without clear justification
*   🚧 Hit a wall in scenarios demanding trustworthy, step-by-step explanations

This limits their use in critical domains. That's where **KG-TRACES** steps in!

<p align="center">
  <img src="assets/teaser.png" width="750" alt="KG-TRACES Teaser Image: Comparison of Reasoning Methods">
</p>
*Figure 1: KG-TRACES (d) stands out by generating faithful, attributable responses, adapting to different KG access conditions.*


## 💡 Our Solution: KG-TRACES

KG-TRACES is a novel framework that explicitly teaches LLMs *how* to reason by supervising their internal "thought process" with knowledge graphs guidance. We guide them to:

1.  🗺️ **Chart the Course**: Predict symbolic **knowledge graph reasoning paths** from question to answer.
2.  📝 **Show Their Work**: Generate **attribution-aware reasoning explanations**, clearly claim whether each step comes from the KG or the LLM's internal knowledge 🧠, and how effective it was!


<p align="center">
  <img src="assets/method.png" width="750" alt="KG-TRACES Method Overview">
</p>
*Figure 2: The KG-TRACES framework*

---

## 🌟 Why KG-TRACES Rocks

*   🔍 **Crystal-Clear Explanations**: Understand *why* the LLM reached its conclusion.
*   🛡️ **Trustworthy & Attributable**: Know the evidence source of each reasoning step.
*   💪 **Robust Performance**: Excels even with limited or no direct KG access during inference.
*   🌍 **Versatile**: Shows strong generalization to specialized fields like medicine.


---


## 💬 Chat cases:
<p align="center">
  <img src="assets/qa_result_case.png" width="750" alt="KG-TRACES Method Overview">
</p>

---

## 🚀 Updates & News
*   **`[2025-06-04]`**:  We opensource KG-TRACES codebase and the training dataset of KG-TRACES.
*   **`[2025-06-03]`**:  arxiv KG-TRACES paper is live! Check it out on [arXiv](https://arxiv.org/abs/2506.00783).

---

## 🛠️ Get Started with KG-TRACES

Ready to dive in? Here's how:

### 1. Prerequisites

Make sure you have:
*   Python 3.12+
*   PyTorch 2.60+
*   🤗 Transformers & Datasets
*   deepspeed 0.16+

### 2. Installation

#### Set up environment:
```bash
git clone https://github.com/Edaizi/KG-TRACES.git
cd KG-TRACES
pip install -r requirements.txt
```


#### 📚 Datasets: The Fuel for KG-TRACES
We've meticulously prepared augmented SFT datasets for WebQSP and CWQ, packed with reasoning paths and augmented reasoning process with source attributions. Find them on Hugging Face:

- [KG-TRACES-WebQSP](https://huggingface.co/datasets/Edaizi/KG-TRACES-WebQSP)
- [KG-TRACES-CWQ](https://huggingface.co/datasets/Edaizi/KG-TRACES-CWQ)


Using the Datasets:
```python
from datasets import load_dataset

webqsp_sft_data = load_dataset("Edaizi/KG-TRACES-WebQSP")
cwq_sft_data = load_dataset("Edaizi/KG-TRACES-CWQ")

print("Example WebQSP SFT instance:")
print(webqsp_sft_data['train'][0]) # Show an example
```



#### 🚂 Training KG-TRACES Model

Just run `scripts/train.sh` easily: 

```bash
bash scripts/train.sh
```


#### 🧪 Inference

Just run `scripts/predict.sh` easily:

```bash
bash scripts/predict.sh
```


#### 🎁 Pretrained Models: Ready to Use!
Don't want to train from scratch? Grab our fine-tuned KG-TRACES models from the Hugging Face Model Hub: [KG-TRACES](https://huggingface.co/Edaizi/KG-TRACES)


``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_hub_name = "Edaizi/KG-TRACES"
tokenizer = AutoTokenizer.from_pretrained(model_hub_name)
model = AutoModelForCausalLM.from_pretrained(model_hub_name)
```


# 📈 Results Highlights

<p align="center">
  <img src="assets/main_result.png" width="750" alt="KG-TRACES Method Overview">
</p>


# 📜 Citation
If KG-TRACES helps your research or project, we'd love a shout-out! Please cite:

```Bibtex
@misc{wu2025kgtracesenhancinglargelanguage,
      title={KG-TRACES: Enhancing Large Language Models with Knowledge Graph-constrained Trajectory Reasoning and Attribution Supervision}, 
      author={Rong Wu and Pinlong Cai and Jianbiao Mei and Licheng Wen and Tao Hu and Xuemeng Yang and Daocheng Fu and Botian Shi},
      year={2025},
      eprint={2506.00783},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.00783}, 
}
```




# 🙏 Acknowledgements
We utilized the following repos during development:
- [Qwen2.5](https://github.com/QwenLM/Qwen3/tree/v2.5)
- [Llama3](https://github.com/meta-llama/llama3)
- [RoG](https://github.com/RManLuo/reasoning-on-graphs)


