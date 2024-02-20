
prompt_lengths=("default" "short" "mistral")

for prompt_length in ${prompt_lengths[@]}; do

#echo """
python train_unlikelihood.py \
    --prompt_length ${prompt_length} \
    --config sampling --pretrained_model_path ${model}
#"""

if [ ${prompt_length} == "default" ]; then
#echo """
# testset, only for harmless
python generate.py \
    --use_sampling --n_samples 25 --use_soft_prompt --use_harmless --use_testset --do_unlikelihood \
    --prompt_length ${prompt_length} --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless --use_testset \
    --model_names ${model_name}_with_soft_unlikelihood_${prompt_length}
#"""
fi

#echo """
# alpaca eval
python generate.py \
    --use_sampling --n_samples 1 --use_soft_prompt --use_alpaca --pretrained_model_path ${model} \
    --prompt_length ${prompt_length} --do_unlikelihood
#"""


# harmful eval for malicious
#echo """
python generate.py \
    --use_sampling --n_samples 25 --use_malicious --pretrained_model_path ${model} \
    --use_soft_prompt --prompt_length ${prompt_length} --do_unlikelihood

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_soft_unlikelihood_${prompt_length}
#"""

#echo """
python generate.py \
    --use_sampling --n_samples 25 --use_advbench --pretrained_model_path ${model} \
    --use_soft_prompt --prompt_length ${prompt_length} --do_unlikelihood

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_soft_unlikelihood_${prompt_length}
#"""

done

