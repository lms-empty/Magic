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
model='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/math_output/deepseek_r1_instruct'
data='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/data/math_kg_yaml.json'
output_dir='/public/home/linsenhao2024/Retrieval-and-Reasoning-on-KGs-main/math_output/deepseek_r1_instruct_graph'
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
echo "Model path exists: $(test -d "$model" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"
echo "Data file exists: $(test -f "$data" && echo "YES" || echo "NO")" | tee -a "$OUTPUT_FILE"
ls -la "$data" 2>&1 | tee -a "$OUTPUT_FILE"

echo "=== MODEL ARCHITECTURE CHECK ===" | tee -a "$OUTPUT_FILE"
python -c "
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained('$model')
    print('Model type:', config.model_type)
    print('Architecture:', config.architectures[0] if hasattr(config, 'architectures') else 'Unknown')
    print('Hidden size:', config.hidden_size if hasattr(config, 'hidden_size') else 'Unknown')
    print('Num attention heads:', config.num_attention_heads if hasattr(config, 'num_attention_heads') else 'Unknown')
    print('Num layers:', config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'Unknown')
except Exception as e:
    print('Error loading model config:', e)
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

# 构造训练参数数组 - 修改为适合两卡一进程的配置，仅使用脚本支持的参数
TRAIN_ARGS=(
    --model_name_or_path "$model"
    --train_file "$data"
    --per_device_train_batch_size 1
    --cutoff_len 128
    --gradient_accumulation_steps 32  # 减少梯度累积步数，因为两卡能处理更多数据
    --per_device_eval_batch_size 1
    --learning_rate 1e-5
    --num_train_epochs 2
    --preprocessing_num_workers 1
    --output_dir "$output_dir"
    --peft
    --kg_pretrain 
    --lora_r 4
    --lora_alpha 8
    --lora_dropout 0.05
    --lora_target_modules q_proj,v_proj
    --checkpointing_steps 100
    --report_to none
    --seed 42
    --trust_remote_code
    --low_cpu_mem_usage
)

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
python "$script_path" --help > /tmp/script_help.txt 2>&1
if grep -q "resume_from_checkpoint" /tmp/script_help.txt; then
    echo "Script supports resume_from_checkpoint" | tee -a "$OUTPUT_FILE"
    if [ -n "$RESUME_CHECKPOINT" ]; then
        TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
        echo "Added resume checkpoint parameter" | tee -a "$OUTPUT_FILE"
    fi
else
    echo "Script does NOT support resume_from_checkpoint" | tee -a "$OUTPUT_FILE"
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
        --model_name_or_path "$model"
        --train_file "$data"
        --per_device_train_batch_size 1
        --cutoff_len 128
        --gradient_accumulation_steps 64  # 增加以补偿单GPU
        --per_device_eval_batch_size 1
        --learning_rate 1e-5
        --num_train_epochs 2
        --preprocessing_num_workers 1
        --output_dir "$output_dir"
        --kg_pretrain 
        --peft
        --lora_r 4
        --lora_alpha 8
        --lora_dropout 0.05
        --lora_target_modules q_proj,v_proj
        --checkpointing_steps 100
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

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "=== DEBUGGING INFO ===" | tee -a "$OUTPUT_FILE"
    if [ -d "$output_dir" ]; then
        echo "Available checkpoints for next resume:" | tee -a "$OUTPUT_FILE"
        find "$output_dir" -maxdepth 1 -type d -name "step_*" | sort -V | tail -5 | tee -a "$OUTPUT_FILE"
    fi
    echo "Check log file: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
fi

exit $TRAIN_EXIT_CODE