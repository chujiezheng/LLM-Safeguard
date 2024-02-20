
system_prompt_type="all"

#echo """
python estimate.py \
    --system_prompt_type ${system_prompt_type} \
    --config sampling --pretrained_model_path ${model}
#"""

prompt_lengths=("default" "short" "mistral")
for prompt_length in ${prompt_lengths[@]}; do

#echo """
python train.py \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} \
    --config sampling --pretrained_model_path ${model}
#"""


if [ ${prompt_length} == "default" ]; then
#echo """
# testset, only for harmless
python generate.py \
    --use_sampling --n_samples 25 --use_soft_prompt --use_harmless --use_testset \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless --use_testset \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length}
#"""
fi


#echo """
# alpaca eval
python generate.py \
    --use_sampling --n_samples 1 --use_soft_prompt --use_alpaca --pretrained_model_path ${model} \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length}
#"""


#echo """
# harmful eval for malicious
python generate.py \
    --use_sampling --n_samples 25 --use_malicious --pretrained_model_path ${model} \
    --use_soft_prompt --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length}

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length}
#"""


#echo """
# harmful eval for advbench
python generate.py \
    --use_sampling --n_samples 25 --use_advbench --pretrained_model_path ${model} \
    --use_soft_prompt --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length}

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length}
#"""


for ablate in "norm" "refu" "harm"; do
#echo """
python train.py \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} --ablate_${ablate} \
    --config sampling --pretrained_model_path ${model}

python generate.py \
    --use_sampling --n_samples 1 --use_soft_prompt --use_alpaca --pretrained_model_path ${model} --ablate_${ablate} \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length}
#"""

#echo """
python generate.py \
    --use_sampling --n_samples 25 --use_malicious --pretrained_model_path ${model} --ablate_${ablate} \
    --use_soft_prompt --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length}

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length}_no${ablate}
#"""

#echo """
python generate.py \
    --use_sampling --n_samples 25 --use_advbench --pretrained_model_path ${model} --ablate_${ablate} \
    --use_soft_prompt --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length}

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length}_no${ablate}
#"""

if [ ${prompt_length} == "default" ]; then
#echo """
# testset, only for harmless
python generate.py \
    --use_sampling --n_samples 25 --use_soft_prompt --use_harmless --use_testset --ablate_${ablate} \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless --use_testset \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length}_no${ablate}
#"""
fi

done

done
