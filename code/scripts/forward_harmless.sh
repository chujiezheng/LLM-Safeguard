full_model_names=(
    "meta-llama/Llama-2-7b-chat-hf"
    "codellama/CodeLlama-7b-Instruct-hf"
    "lmsys/vicuna-7b-v1.5"
    "microsoft/Orca-2-7b"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "openchat/openchat-3.5"
    "openchat/openchat-3.5-1210"
)

for full_model_name in ${full_model_names[@]}; do

model=${HF_MODELS}/${full_model_name}

python forward.py \
    --use_harmless \
    --pretrained_model_path ${model}

echo """
python forward_with_soft.py \
    --use_testset \
    --pretrained_model_path ${model}

python forward.py \
    --use_testset \
    --pretrained_model_path ${model}
#"""

done
