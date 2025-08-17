#!/bin/bash
#SBATCH --job-name=deepseek_train
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:2
#SBATCH --output=logs/deepseek_train_%j.out

# 获取任务ID和节点名
JOB_ID=$SLURM_JOB_ID
NODE_NAME=$(hostname)
OUTPUT_FILE="logs/deepseek_train_${JOB_ID}.log"

# 创建日志目录
mkdir -p logs

echo "=== JOB START INFO ===" | tee -a "$OUTPUT_FILE"
echo "Job started at: $(date)" | tee -a "$OUTPUT_FILE"
echo "Running on node: $NODE_NAME" | tee -a "$OUTPUT_FILE"
echo "Job ID: $JOB_ID" | tee -a "$OUTPUT_FILE"

# 设置训练参数 
base_model='/public/home/linsenhao2024/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
existing_lora='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/math_output/deepseek_r1_instruct_graph'
data='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/data/kg_reason.json'
output_dir='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/math_output/deepseek_r1_instruct_graph_qa'
script_path='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/run_clm_finetune.py'

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mathKG

echo "=== ENVIRONMENT CHECK ===" | tee -a "$OUTPUT_FILE"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())" | tee -a "$OUTPUT_FILE"
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" | tee -a "$OUTPUT_FILE"
python -c "import transformers; print('Transformers version:', transformers.__version__)" | tee -a "$OUTPUT_FILE"

# 内存优化 - 关键修改：配置两卡一进程
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# 新增：启用模型并行而非数据并行
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0

echo "=== SYSTEM RESOURCES ===" | tee -a "$OUTPUT_FILE"
free -h | tee -a "$OUTPUT_FILE"
nvidia-smi | tee -a "$OUTPUT_FILE"

echo "=== FILE CHECK ===" | tee -a "$OUTPUT_FILE"
echo "Base model path exists: $(test -d "$base_model" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"
echo "Existing LoRA path exists: $(test -d "$existing_lora" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"
echo "Data file exists: $(test -f "$data" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"
ls -la "$data" 2>&1 | tee -a "$OUTPUT_FILE"

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
except Exception as e:
    print('Error loading base model config:', e)
" | tee -a "$OUTPUT_FILE"

echo "=== EXISTING LORA CHECK ===" | tee -a "$OUTPUT_FILE"
python -c "
import json
import os
lora_config_path = '$existing_lora/adapter_config.json'
if os.path.exists(lora_config_path):
    with open(lora_config_path, 'r') as f:
        config = json.load(f)
    print('Existing LoRA config:')
    print('  PEFT type:', config.get('peft_type', 'Unknown'))
    print('  Task type:', config.get('task_type', 'Unknown'))
    print('  LoRA rank (r):', config.get('r', 'Unknown'))
    print('  LoRA alpha:', config.get('lora_alpha', 'Unknown'))
    print('  LoRA dropout:', config.get('lora_dropout', 'Unknown'))
    print('  Target modules:', config.get('target_modules', 'Unknown'))
    print('  Base model:', config.get('base_model_name_or_path', 'Unknown'))
else:
    print('No existing LoRA config found at:', lora_config_path)
    print('Available files in LoRA directory:')
    if os.path.exists('$existing_lora'):
        for f in os.listdir('$existing_lora'):
            print(' ', f)
" | tee -a "$OUTPUT_FILE"

echo "=== CHECKPOINT DETECTION ===" | tee -a "$OUTPUT_FILE"
RESUME_CHECKPOINT=""
if [ -d "$output_dir" ]; then
    echo "Output directory exists, checking for checkpoints..." | tee -a "$OUTPUT_FILE"
    
    # 清理可能损坏的检查点
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
        else
            echo "Found checkpoint directory but missing model files: $LATEST_CHECKPOINT" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo "No valid checkpoint directories found" | tee -a "$OUTPUT_FILE"
    fi
else
    echo "Output directory does not exist: $output_dir" | tee -a "$OUTPUT_FILE"
fi

# 构造训练参数数组 - 修改为使用基础模型 + 现有LoRA
TRAIN_ARGS=(
    --model_name_or_path "$base_model"
    --train_file "$data"
    --per_device_train_batch_size 1
    --cutoff_len 128
    --gradient_accumulation_steps 32
    --per_device_eval_batch_size 1
    --learning_rate 1e-5
    --num_train_epochs 2
    --preprocessing_num_workers 1
    --output_dir "$output_dir"
    --peft
    --lora_r 4
    --lora_alpha 8
    --lora_dropout 0.05
    --lora_target_modules q_proj,v_proj
    --checkpointing_steps 39416
    --report_to none
    --seed 42
    --trust_remote_code
    --low_cpu_mem_usage
)

# 检查脚本是否支持加载现有LoRA适配器
echo "=== CHECKING SCRIPT LoRA SUPPORT ===" | tee -a "$OUTPUT_FILE"
python "$script_path" --help > /tmp/script_help.txt 2>&1
if grep -q "adapter_name_or_path\|peft_model_id\|lora_model_path" /tmp/script_help.txt; then
    echo "Script supports loading existing LoRA adapters" | tee -a "$OUTPUT_FILE"
    # 根据具体支持的参数名添加
    if grep -q "adapter_name_or_path" /tmp/script_help.txt; then
        TRAIN_ARGS+=(--adapter_name_or_path "$existing_lora")
        echo "Added --adapter_name_or_path parameter" | tee -a "$OUTPUT_FILE"
    elif grep -q "peft_model_id" /tmp/script_help.txt; then
        TRAIN_ARGS+=(--peft_model_id "$existing_lora")
        echo "Added --peft_model_id parameter" | tee -a "$OUTPUT_FILE"
    elif grep -q "lora_model_path" /tmp/script_help.txt; then
        TRAIN_ARGS+=(--lora_model_path "$existing_lora")
        echo "Added --lora_model_path parameter" | tee -a "$OUTPUT_FILE"
    fi
else
    echo "Script does NOT support loading existing LoRA adapters" | tee -a "$OUTPUT_FILE"
    echo "Will create new LoRA from scratch" | tee -a "$OUTPUT_FILE"
fi

# 检查点恢复说明
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "=== CHECKPOINT FOUND BUT RESUME NOT SUPPORTED ===" | tee -a "$OUTPUT_FILE"
    echo "Found checkpoint: $RESUME_CHECKPOINT" | tee -a "$OUTPUT_FILE"
    echo "Note: Current script does not support --resume_from_checkpoint" | tee -a "$OUTPUT_FILE"
    echo "Training will start from beginning but existing checkpoints are preserved" | tee -a "$OUTPUT_FILE"
else
    echo "=== STARTING NEW TRAINING ===" | tee -a "$OUTPUT_FILE"
fi

# 首先检查脚本支持的参数
echo "=== CHECKING SCRIPT PARAMETERS ===" | tee -a "$OUTPUT_FILE"
if grep -q "resume_from_checkpoint" /tmp/script_help.txt; then
    echo "Script supports resume_from_checkpoint" | tee -a "$OUTPUT_FILE"
    if [ -n "$RESUME_CHECKPOINT" ]; then
        TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
        echo "Added resume checkpoint parameter" | tee -a "$OUTPUT_FILE"
    fi
else
    echo "Script does NOT support resume_from_checkpoint" | tee -a "$OUTPUT_FILE"
fi

# 如果脚本不支持直接加载LoRA，创建临时合并脚本
if ! grep -q "adapter_name_or_path\|peft_model_id\|lora_model_path" /tmp/script_help.txt; then
    echo "=== CREATING LORA MERGE SCRIPT ===" | tee -a "$OUTPUT_FILE"
    cat > /tmp/merge_lora.py << 'EOF'
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

def merge_and_save_lora(base_model_path, lora_adapter_path, output_path):
    """合并基础模型和LoRA适配器，保存为新的基础模型"""
    print(f"Loading base model from: {base_model_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("Merging LoRA weights...")
    
    # 合并LoRA权重到基础模型
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    
    # 保存合并后的模型
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    # 复制tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge completed successfully!")
    
    # 清理内存
    del base_model, model, merged_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_lora.py <base_model_path> <lora_adapter_path> <output_path>")
        sys.exit(1)
    
    base_model_path = sys.argv[1]
    lora_adapter_path = sys.argv[2]  
    output_path = sys.argv[3]
    
    merge_and_save_lora(base_model_path, lora_adapter_path, output_path)
EOF

    # 创建合并后的模型路径
    merged_model_path="/tmp/merged_deepseek_r1_$(date +%s)"
    
    echo "=== MERGING EXISTING LORA WITH BASE MODEL ===" | tee -a "$OUTPUT_FILE"
    echo "Merged model will be saved to: $merged_model_path" | tee -a "$OUTPUT_FILE"
    
    python /tmp/merge_lora.py "$base_model" "$existing_lora" "$merged_model_path" 2>&1 | tee -a "$OUTPUT_FILE"
    
    if [ $? -eq 0 ] && [ -d "$merged_model_path" ]; then
        echo "LoRA merge successful, using merged model for training" | tee -a "$OUTPUT_FILE"
        # 更新训练参数使用合并后的模型
        TRAIN_ARGS[1]="$merged_model_path"  # 替换 --model_name_or_path 的值
        echo "Updated model path to: $merged_model_path" | tee -a "$OUTPUT_FILE"
    else
        echo "LoRA merge failed, falling back to base model only" | tee -a "$OUTPUT_FILE"
    fi
fi

# 启动训练 - 关键修改：使用单进程多GPU配置
echo "=== TRAINING ATTEMPT 1 - SINGLE PROCESS WITH 2 GPUS ===" | tee -a "$OUTPUT_FILE"
echo "Using DataParallel mode with 2 GPUs in single process" | tee -a "$OUTPUT_FILE"
echo "Command: python $script_path ${TRAIN_ARGS[*]}" | tee -a "$OUTPUT_FILE"

# 直接使用python运行，让PyTorch自动检测并使用两张GPU
python "$script_path" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_FILE"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=== TRAINING ATTEMPT 1 FAILED ===" | tee -a "$OUTPUT_FILE"
    echo "Exit code: $TRAIN_EXIT_CODE" | tee -a "$OUTPUT_FILE"
    
    # 尝试使用accelerate但单进程配置
    echo "=== TRAINING ATTEMPT 2 - ACCELERATE SINGLE PROCESS ===" | tee -a "$OUTPUT_FILE"
    
    echo "Command: accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 $script_path ${TRAIN_ARGS[*]}" | tee -a "$OUTPUT_FILE"
    
    accelerate launch \
        --num_processes=1 \
        --num_machines=1 \
        --mixed_precision=bf16 \
        "$script_path" \
        "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_FILE"
    TRAIN_EXIT_CODE=$?
fi

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=== TRAINING ATTEMPT 2 FAILED ===" | tee -a "$OUTPUT_FILE"
    echo "Exit code: $TRAIN_EXIT_CODE" | tee -a "$OUTPUT_FILE"
    
    # 最后尝试单GPU
    echo "=== TRAINING ATTEMPT 3 - SINGLE GPU FALLBACK ===" | tee -a "$OUTPUT_FILE"
    export CUDA_VISIBLE_DEVICES=0
    
    # 调整批次大小以适应单GPU，仅使用脚本支持的参数
    TRAIN_ARGS_SINGLE=(
        --model_name_or_path "${TRAIN_ARGS[1]}"  # 使用已确定的模型路径
        --train_file "$data"
        --per_device_train_batch_size 1
        --cutoff_len 128
        --gradient_accumulation_steps 64  # 增加以补偿单GPU
        --per_device_eval_batch_size 1
        --learning_rate 1e-5
        --num_train_epochs 2
        --preprocessing_num_workers 1
        --output_dir "$output_dir"
        --peft
        --lora_r 4
        --lora_alpha 8
        --lora_dropout 0.05
        --lora_target_modules q_proj,v_proj
        --checkpointing_steps 39416
        --report_to none
        --seed 42
        --trust_remote_code
        --low_cpu_mem_usage
    )
    
    echo "Command: python $script_path ${TRAIN_ARGS_SINGLE[*]}" | tee -a "$OUTPUT_FILE"
    python "$script_path" "${TRAIN_ARGS_SINGLE[@]}" 2>&1 | tee -a "$OUTPUT_FILE"
    TRAIN_EXIT_CODE=$?
fi

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== TRAINING COMPLETED SUCCESSFULLY ===" | tee -a "$OUTPUT_FILE"
else
    echo "=== TRAINING FAILED ===" | tee -a "$OUTPUT_FILE"
    echo "Final exit code: $TRAIN_EXIT_CODE" | tee -a "$OUTPUT_FILE"
fi

echo "=== FINAL RESOURCE CHECK ===" | tee -a "$OUTPUT_FILE"
nvidia-smi | tee -a "$OUTPUT_FILE"
echo "Job ended at: $(date)" | tee -a "$OUTPUT_FILE"

# 清理临时文件
if [ -d "/tmp/merged_deepseek_r1_"* ]; then
    echo "=== CLEANUP ===" | tee -a "$OUTPUT_FILE"
    echo "Cleaning up temporary merged model..." | tee -a "$OUTPUT_FILE"
    rm -rf /tmp/merged_deepseek_r1_*
    echo "Cleanup completed" | tee -a "$OUTPUT_FILE"
fi

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=== DEBUGGING INFO ===" | tee -a "$OUTPUT_FILE"
    if [ -d "$output_dir" ]; then
        echo "Available checkpoints for next resume:" | tee -a "$OUTPUT_FILE"
        find "$output_dir" -maxdepth 1 -type d -name "step_*" | sort -V | tail -5 | tee -a "$OUTPUT_FILE"
    fi
    echo "Check log file: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
fi

exit $TRAIN_EXIT_CODE