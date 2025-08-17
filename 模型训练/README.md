# README.md

## Step 1: Base Model Fine-tuning
Fine-tune the base DeepSeek model on instruction data.
```bash
bash train_deepseek.sbt
```
This script initializes the base model fine-tuning with LoRA (Low-Rank Adaptation) for efficient parameter updates. Key configurations include:
- Base model: `DeepSeek-R1-Distill-Qwen-7B`
- Training data: `math_kg_yaml.json`
- Output directory: `deepseek_r1_instruct`
- LoRA parameters: r=4, alpha=8, dropout=0.05


## Step 2: Knowledge Graph Pre-training
Enhance the model with knowledge graph structure understanding.
```bash
bash train_deepseek_kg.sbt
```
This step enables knowledge graph-aware pre-training with the `--kg_pretrain` flag, focusing on:
- Continued training from the base fine-tuned model
- KG-specific data: `math_kg_yaml.json`
- Output directory: `deepseek_r1_instruct_graph`
- Preserves LoRA efficiency while incorporating graph structure knowledge


## Step 3: Knowledge Question Answering Fine-tuning
Adapt the model for knowledge graph-based question answering tasks.
```bash
bash train_deepseek_kg_qa.sbt
```
Key features of this step:
- Fine-tunes on KGQA data: `kg_reason.json`
- Builds on KG-pre-trained model
- Output directory: `deepseek_r1_instruct_graph_qa`
- Optimized for multi-hop reasoning tasks


## Step 4: Learning Path Planning Fine-tuning
Specialize the model for educational path recommendation.
```bash
bash train_deepseek_kg_qa_ver.sbt
```
This advanced training step:
- Focuses on learning path data: `train.jsonl`/`eval.jsonl`
- Integrates multi-hop reasoning capabilities
- Output directory: `deepseek_r1_learning_path`
- Adds capabilities for dependency analysis and concept sequencing



