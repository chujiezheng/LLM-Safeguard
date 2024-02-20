
#echo """
# base
python generate.py \
    --use_sampling --n_samples 25 \
    --use_default_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_default

python generate.py \
    --use_sampling --n_samples 25 \
    --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}

python generate.py \
    --use_sampling --n_samples 25 \
    --use_short_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_short

python generate.py \
    --use_sampling --n_samples 25 \
    --use_mistral_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_mistral
#"""

#echo """
# malicious
python generate.py \
    --use_sampling --n_samples 25 --use_malicious \
    --use_default_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_default

python generate.py \
    --use_sampling --n_samples 25 --use_malicious \
    --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}

python generate.py \
    --use_sampling --n_samples 25 --use_malicious \
    --use_short_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_short

python generate.py \
    --use_sampling --n_samples 25 --use_malicious \
    --use_mistral_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_mistral
#"""

#echo """
# advbench
python generate.py \
    --use_sampling --n_samples 25 --use_advbench \
    --use_default_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_default

python generate.py \
    --use_sampling --n_samples 25 --use_advbench \
    --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}

python generate.py \
    --use_sampling --n_samples 25 --use_advbench \
    --use_short_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_short

python generate.py \
    --use_sampling --n_samples 25 --use_advbench \
    --use_mistral_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/meta-llama/LlamaGuard-7b \
    --model_names ${model_name}_with_mistral
#"""
