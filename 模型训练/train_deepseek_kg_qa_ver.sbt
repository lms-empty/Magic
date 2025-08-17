#!/bin/bash
#SBATCH --job-name=learning_path_train
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:4
#SBATCH --output=logs/learning_path_train_%j.out


# 获取任务ID和节点名
JOB_ID=$SLURM_JOB_ID
NODE_NAME=$(hostname)
OUTPUT_FILE="logs/learning_path_train_${JOB_ID}.log"

# 创建日志目录
mkdir -p logs

echo "=== LEARNING PATH TRAINING JOB START ===" | tee -a "$OUTPUT_FILE"
echo "Job started at: $(date)" | tee -a "$OUTPUT_FILE"
echo "Running on node: $NODE_NAME" | tee -a "$OUTPUT_FILE"
echo "Job ID: $JOB_ID" | tee -a "$OUTPUT_FILE"

# =================== 配置路径参数 ===================
# 基础模型路径
base_model='/public/home/linsenhao2024/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

# 现有的多跳推理LoRA适配器（作为基础）
existing_lora='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/math_output/deepseek_r1_instruct_graph_qa'

# 学习路径规划训练数据（使用转换后的数据）
train_data='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/training_data/train.jsonl'
eval_data='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/training_data/eval.jsonl'

# 输出目录（学习路径规划模型）
output_dir='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/math_output/deepseek_r1_learning_path'

# 训练脚本路径
script_path='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/run_clm_finetune.py'

# =================== 环境设置 ===================
# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mathKG

echo "=== ENVIRONMENT CHECK ===" | tee -a "$OUTPUT_FILE"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())" | tee -a "$OUTPUT_FILE"
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" | tee -a "$OUTPUT_FILE"
python -c "import transformers; print('Transformers version:', transformers.__version__)" | tee -a "$OUTPUT_FILE"
python -c "import peft; print('PEFT version:', peft.__version__)" | tee -a "$OUTPUT_FILE"

# GPU内存优化配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0

echo "=== SYSTEM RESOURCES ===" | tee -a "$OUTPUT_FILE"
free -h | tee -a "$OUTPUT_FILE"
nvidia-smi | tee -a "$OUTPUT_FILE"

# =================== 数据验证 ===================
echo "=== TRAINING DATA VALIDATION ===" | tee -a "$OUTPUT_FILE"
echo "Base model path exists: $(test -d "$base_model" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"
echo "Existing LoRA path exists: $(test -d "$existing_lora" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"
echo "Train data exists: $(test -f "$train_data" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"
echo "Eval data exists: $(test -f "$eval_data" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"

# 检查训练数据格式和数量
if [ -f "$train_data" ]; then
    train_count=$(wc -l < "$train_data")
    echo "Training samples: $train_count" | tee -a "$OUTPUT_FILE"
    echo "Sample training data (first 3 lines):" | tee -a "$OUTPUT_FILE"
    head -3 "$train_data" | tee -a "$OUTPUT_FILE"
fi

if [ -f "$eval_data" ]; then
    eval_count=$(wc -l < "$eval_data")
    echo "Evaluation samples: $eval_count" | tee -a "$OUTPUT_FILE"
fi

# =================== 模型架构验证 ===================
echo "=== BASE MODEL ARCHITECTURE CHECK ===" | tee -a "$OUTPUT_FILE"
python -c "
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained('$base_model', trust_remote_code=True)
    print('Model type:', config.model_type)
    print('Architecture:', config.architectures[0] if hasattr(config, 'architectures') else 'Unknown')
    print('Hidden size:', config.hidden_size if hasattr(config, 'hidden_size') else 'Unknown')
    print('Num attention heads:', config.num_attention_heads if hasattr(config, 'num_attention_heads') else 'Unknown')
    print('Num layers:', config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'Unknown')
    print('Vocabulary size:', config.vocab_size if hasattr(config, 'vocab_size') else 'Unknown')
except Exception as e:
    print('Error loading base model config:', e)
" | tee -a "$OUTPUT_FILE"

# =================== 现有LoRA检查 ===================
echo "=== EXISTING LORA ADAPTER CHECK ===" | tee -a "$OUTPUT_FILE"
python -c "
import json
import os
lora_config_path = '$existing_lora/adapter_config.json'
if os.path.exists(lora_config_path):
    with open(lora_config_path, 'r') as f:
        config = json.load(f)
    print('Existing LoRA configuration:')
    print('  PEFT type:', config.get('peft_type', 'Unknown'))
    print('  Task type:', config.get('task_type', 'Unknown'))
    print('  LoRA rank (r):', config.get('r', 'Unknown'))
    print('  LoRA alpha:', config.get('lora_alpha', 'Unknown'))
    print('  LoRA dropout:', config.get('lora_dropout', 'Unknown'))
    print('  Target modules:', config.get('target_modules', 'Unknown'))
    print('  Base model:', config.get('base_model_name_or_path', 'Unknown'))
else:
    print('No existing LoRA config found at:', lora_config_path)
    if os.path.exists('$existing_lora'):
        print('Available files in LoRA directory:')
        for f in os.listdir('$existing_lora'):
            print(' ', f)
" | tee -a "$OUTPUT_FILE"

# =================== 检查点检测和清理 ===================
echo "=== CHECKPOINT DETECTION ===" | tee -a "$OUTPUT_FILE"
RESUME_CHECKPOINT=""
if [ -d "$output_dir" ]; then
    echo "Output directory exists, checking for checkpoints..." | tee -a "$OUTPUT_FILE"
    
    # 清理损坏的检查点
    echo "Cleaning corrupted checkpoints..." | tee -a "$OUTPUT_FILE"
    find "$output_dir" -maxdepth 1 -type d -name "step_*" | while read checkpoint_dir; do
        if [ ! -f "$checkpoint_dir/pytorch_model.bin" ] && [ ! -f "$checkpoint_dir/adapter_model.bin" ] && [ ! -d "$checkpoint_dir/global_step*" ]; then
            echo "Removing corrupted checkpoint: $checkpoint_dir" | tee -a "$OUTPUT_FILE"
            rm -rf "$checkpoint_dir"
        fi
    done
    
    LATEST_CHECKPOINT=$(find "$output_dir" -maxdepth 1 -type d -name "step_*" | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ] && [ -d "$LATEST_CHECKPOINT" ]; then
        if [ -f "$LATEST_CHECKPOINT/pytorch_model.bin" ] || [ -f "$LATEST_CHECKPOINT/adapter_model.bin" ] || [ -d "$LATEST_CHECKPOINT/global_step*" ]; then
            RESUME_CHECKPOINT="$LATEST_CHECKPOINT"
            echo "Found valid checkpoint: $RESUME_CHECKPOINT" | tee -a "$OUTPUT_FILE"
            ls -la "$LATEST_CHECKPOINT" | tee -a "$OUTPUT_FILE"
        fi
    fi
else
    echo "Creating new output directory: $output_dir" | tee -a "$OUTPUT_FILE"
    mkdir -p "$output_dir"
fi

# =================== LoRA合并脚本（如果需要） ===================
echo "=== CREATING LORA MERGE UTILITY ===" | tee -a "$OUTPUT_FILE"
cat > /tmp/merge_lora_learning_path.py << 'EOF'
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

def merge_and_save_lora(base_model_path, lora_adapter_path, output_path):
    """合并基础模型和LoRA适配器，为学习路径规划任务准备模型"""
    print(f"Loading base model from: {base_model_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"Loading LoRA adapter (multi-hop reasoning) from: {lora_adapter_path}")
    
    # 加载现有的多跳推理LoRA适配器
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("Merging LoRA weights for learning path foundation...")
    
    # 合并LoRA权重到基础模型
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    
    # 保存合并后的模型
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    # 复制tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge completed! Ready for learning path training.")
    
    # 清理内存
    del base_model, model, merged_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_lora_learning_path.py <base_model_path> <lora_adapter_path> <output_path>")
        sys.exit(1)
    
    base_model_path = sys.argv[1]
    lora_adapter_path = sys.argv[2]  
    output_path = sys.argv[3]
    
    merge_and_save_lora(base_model_path, lora_adapter_path, output_path)
EOF

# =================== 训练参数配置 ===================
echo "=== CONFIGURING TRAINING PARAMETERS ===" | tee -a "$OUTPUT_FILE"

# 检查脚本支持的参数
python "$script_path" --help > /tmp/script_help.txt 2>&1

# 使用现有LoRA作为起点的合并模型路径
merged_model_path="/tmp/merged_deepseek_learning_path_$(date +%s)"

echo "=== MERGING EXISTING MULTI-HOP LORA FOR LEARNING PATH BASE ===" | tee -a "$OUTPUT_FILE"
echo "Creating foundation model with multi-hop reasoning capabilities..." | tee -a "$OUTPUT_FILE"

python /tmp/merge_lora_learning_path.py "$base_model" "$existing_lora" "$merged_model_path" 2>&1 | tee -a "$OUTPUT_FILE"

# 确定使用的模型路径
if [ $? -eq 0 ] && [ -d "$merged_model_path" ]; then
    echo "✅ LoRA merge successful, using merged model as foundation for learning path training" | tee -a "$OUTPUT_FILE"
    MODEL_PATH="$merged_model_path"
else
    echo "⚠️ LoRA merge failed, using base model only" | tee -a "$OUTPUT_FILE"
    MODEL_PATH="$base_model"
fi

# 学习路径规划专用训练参数
TRAIN_ARGS=(
    --model_name_or_path "$MODEL_PATH"
    --train_file "$train_data"
    --validation_file "$eval_data"
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --cutoff_len 128                           # 学习路径对话通常较长
    --gradient_accumulation_steps 32           # 保持有效批次大小
    --learning_rate 2e-5                       # 学习路径规划适中学习率
    --num_train_epochs 3                       # 学习路径规划epochs
    --preprocessing_num_workers 4
    --output_dir "$output_dir"
    --peft                                      # 使用LoRA微调
    --lora_r 4                                  # 增加LoRA rank以支持复杂推理
    --lora_alpha 8                            # 相应调整alpha
    --lora_dropout 0.05                        # 较低dropout保持学习能力
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj  # DeepSeek全面coverage
    --checkpointing_steps 29172                  # 更频繁的检查点保存
    --report_to none
    --seed 42
    --trust_remote_code
    --low_cpu_mem_usage
    --weight_decay 0.01                        # 轻微正则化
    --lr_scheduler_type cosine                 # 余弦学习率调度
    --num_warmup_steps 100                     # 预热步数
    --with_tracking                            # 启用训练跟踪
)

# 添加检查点恢复支持（如果支持）
if grep -q "resume_from_checkpoint" /tmp/script_help.txt && [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
    echo "✅ Added resume checkpoint: $RESUME_CHECKPOINT" | tee -a "$OUTPUT_FILE"
fi

echo "=== TRAINING CONFIGURATION SUMMARY ===" | tee -a "$OUTPUT_FILE"
echo "Model path: $MODEL_PATH" | tee -a "$OUTPUT_FILE"
echo "Train data: $train_data ($train_count samples)" | tee -a "$OUTPUT_FILE"
echo "Eval data: $eval_data ($eval_count samples)" | tee -a "$OUTPUT_FILE"
echo "Output directory: $output_dir" | tee -a "$OUTPUT_FILE"
echo "LoRA configuration: r=8, alpha=16, dropout=0.05" | tee -a "$OUTPUT_FILE"
echo "Learning rate: 2e-5 with cosine schedule" | tee -a "$OUTPUT_FILE"
echo "Effective batch size: $(( 2 * 2 * 16 )) (per_device=2, gpus=2, accumulation=16)" | tee -a "$OUTPUT_FILE"

# =================== 开始训练 ===================
echo "=== STARTING LEARNING PATH TRAINING ===" | tee -a "$OUTPUT_FILE"
echo "Training for learning path planning and recommendation..." | tee -a "$OUTPUT_FILE"
echo "Command: python $script_path ${TRAIN_ARGS[*]}" | tee -a "$OUTPUT_FILE"

# 尝试1: 直接使用python运行，利用DataParallel
echo "=== TRAINING ATTEMPT 1 - SINGLE PROCESS MULTI-GPU ===" | tee -a "$OUTPUT_FILE"
python "$script_path" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_FILE"
TRAIN_EXIT_CODE=$?

# 尝试2: 如果失败，使用accelerate
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=== TRAINING ATTEMPT 2 - ACCELERATE MULTI-GPU ===" | tee -a "$OUTPUT_FILE"
    
    accelerate launch \
        --num_processes=4 \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --use_deepspeed \
        "$script_path" \
        "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_FILE"
    TRAIN_EXIT_CODE=$?
fi

# 尝试3: 单GPU fallback
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=== TRAINING ATTEMPT 3 - SINGLE GPU FALLBACK ===" | tee -a "$OUTPUT_FILE"
    export CUDA_VISIBLE_DEVICES=0
    
    # 调整单GPU参数
    TRAIN_ARGS_SINGLE=(
        --model_name_or_path "$MODEL_PATH"
        --train_file "$train_data"
        --validation_file "$eval_data"
        --per_device_train_batch_size 1
        --per_device_eval_batch_size 1
        --cutoff_len 128
        --gradient_accumulation_steps 32      # 增加以补偿单GPU
        --learning_rate 2e-5
        --num_train_epochs 3
        --preprocessing_num_workers 1
        --output_dir "$output_dir"
        --peft
        --lora_r 4
        --lora_alpha 8
        --lora_dropout 0.05
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
        --checkpointing_steps 29172
        --report_to none
        --seed 42
        --trust_remote_code
        --low_cpu_mem_usage
        --weight_decay 0.01
        --lr_scheduler_type cosine
        --num_warmup_steps 100
        --with_tracking
    )
    
    echo "Command: python $script_path ${TRAIN_ARGS_SINGLE[*]}" | tee -a "$OUTPUT_FILE"
    python "$script_path" "${TRAIN_ARGS_SINGLE[@]}" 2>&1 | tee -a "$OUTPUT_FILE"
    TRAIN_EXIT_CODE=$?
fi

# =================== 训练结果处理 ===================
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== LEARNING PATH TRAINING COMPLETED SUCCESSFULLY ===" | tee -a "$OUTPUT_FILE"
    echo "✅ Model trained for learning path planning and recommendation" | tee -a "$OUTPUT_FILE"
    
    # 保存训练摘要
    echo "=== TRAINING SUMMARY ===" | tee -a "$OUTPUT_FILE"
    echo "Task: Learning Path Planning and Recommendation" | tee -a "$OUTPUT_FILE"
    echo "Base capabilities: Multi-hop reasoning (from existing LoRA)" | tee -a "$OUTPUT_FILE"
    echo "New capabilities: Dependency analysis, concept guidance, path recommendation" | tee -a "$OUTPUT_FILE"
    echo "Final model location: $output_dir" | tee -a "$OUTPUT_FILE"
    
    # 检查生成的模型文件
    if [ -d "$output_dir" ]; then
        echo "Generated model files:" | tee -a "$OUTPUT_FILE"
        ls -la "$output_dir" | tee -a "$OUTPUT_FILE"
    fi
    
else
    echo "=== LEARNING PATH TRAINING FAILED ===" | tee -a "$OUTPUT_FILE"
    echo "❌ Exit code: $TRAIN_EXIT_CODE" | tee -a "$OUTPUT_FILE"
    
    # 调试信息
    echo "=== DEBUGGING INFORMATION ===" | tee -a "$OUTPUT_FILE"
    if [ -d "$output_dir" ]; then
        echo "Available checkpoints for recovery:" | tee -a "$OUTPUT_FILE"
        find "$output_dir" -maxdepth 1 -type d -name "step_*" | sort -V | tail -5 | tee -a "$OUTPUT_FILE"
    fi
fi

# =================== 资源清理 ===================
echo "=== FINAL RESOURCE CHECK ===" | tee -a "$OUTPUT_FILE"
nvidia-smi | tee -a "$OUTPUT_FILE"
echo "Job ended at: $(date)" | tee -a "$OUTPUT_FILE"

# 清理临时合并模型
if [ -d "$merged_model_path" ]; then
    echo "=== CLEANUP ===" | tee -a "$OUTPUT_FILE"
    echo "Cleaning up temporary merged model: $merged_model_path" | tee -a "$OUTPUT_FILE"
    rm -rf "$merged_model_path"
    echo "Cleanup completed" | tee -a "$OUTPUT_FILE"
fi

# =================== 训练完成后的验证脚本 ===================
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== CREATING MODEL VALIDATION SCRIPT ===" | tee -a "$OUTPUT_FILE"
    cat > "$output_dir/validate_learning_path_model.py" << 'EOF'
#!/usr/bin/env python3
"""
学习路径规划模型验证脚本
验证训练后的模型在学习路径规划任务上的表现
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_model(base_model_path, lora_path):
    """加载训练后的学习路径模型"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model, tokenizer

def test_learning_path_capabilities(model, tokenizer):
    """测试学习路径规划能力"""
    
    test_cases = [
        "分析柯西序列和实数定义之间的学习依赖关系。",
        "我已经掌握了0.999...，接下来应该学习什么？",
        "如何学习和理解无限循环小数？"
    ]
    
    print("=== 学习路径规划模型能力测试 ===")
    
    for i, instruction in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {instruction}")
        
        # 构造输入
        prompt = f"指令：{instruction}\n回应："
        
        # 编码
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        print(f"模型回应: {response}")
        print("-" * 80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python validate_learning_path_model.py <base_model_path> <lora_path>")
        sys.exit(1)
    
    base_model_path = sys.argv[1]
    lora_path = sys.argv[2]
    
    print("加载学习路径规划模型...")
    model, tokenizer = load_model(base_model_path, lora_path)
    
    print("开始能力测试...")
    test_learning_path_capabilities(model, tokenizer)
    
    print("\n=== 验证完成 ===")
EOF

    chmod +x "$output_dir/validate_learning_path_model.py"
    echo "✅ Created model validation script: $output_dir/validate_learning_path_model.py" | tee -a "$OUTPUT_FILE"
    echo "To validate the model, run:" | tee -a "$OUTPUT_FILE"
    echo "python $output_dir/validate_learning_path_model.py $base_model $output_dir" | tee -a "$OUTPUT_FILE"
fi

echo "=== LEARNING PATH TRAINING JOB COMPLETED ===" | tee -a "$OUTPUT_FILE"
echo "Check full log at: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"

exit $TRAIN_EXIT_CODE