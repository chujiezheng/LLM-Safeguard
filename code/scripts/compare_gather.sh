
model_names=(
    "meta-llama/Llama-2-7b-chat-hf"
    "lmsys/vicuna-7b-v1.5"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "openchat/openchat-3.5"
    "codellama/CodeLlama-7b-Instruct-hf"
    "microsoft/Orca-2-7b"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "openchat/openchat-3.5-1210"
)
full_model_names=()
for model_name in ${model_names[@]}; do
    full_model_names+=("${HF_MODELS}/${model_name}")
done

for system_prompt_type in "all"; do

python compare_pca_soft_harmfulness.py \
    --pretrained_model_paths ${full_model_names[@]} \
    --config sampling \
    --system_prompt_type ${system_prompt_type} \
    --output_path comparisons/pca_soft

python compare_pca_harmfulness_boundary.py \
    --pretrained_model_paths ${full_model_names[@]} \
    --config sampling \
    --system_prompt_type ${system_prompt_type} \
    --output_path comparisons/pca

echo """
python compare_pca_soft_refusal.py \
    --pretrained_model_paths ${full_model_names[@]} \
    --config sampling \
    --system_prompt_type ${system_prompt_type} \
    --output_path comparisons/pca_soft

python compare_pca_refusal_boundary.py \
    --pretrained_model_paths ${full_model_names[@]} \
    --config sampling \
    --system_prompt_type ${system_prompt_type} \
    --output_path comparisons/pca
#"""

done
