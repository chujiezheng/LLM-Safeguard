full_model_names=(
    "mistralai/Mistral-7B-Instruct-v0.1"
)

for full_model_name in ${full_model_names[@]}; do
model=${HF_MODELS}/${full_model_name}
model_name=$(basename ${full_model_name})

source scripts/train_meta.sh
#source scripts/train_meta_unlikelihood.sh

done
