
#echo """
# base
python generate.py \
    --use_sampling --n_samples 25 --use_harmless \
    --use_default_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless \
    --model_names ${model_name}_with_default

python generate.py \
    --use_sampling --n_samples 25 --use_harmless \
    --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless \
    --model_names ${model_name}

python generate.py \
    --use_sampling --n_samples 25 --use_harmless \
    --use_short_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless \
    --model_names ${model_name}_with_short

python generate.py \
    --use_sampling --n_samples 25 --use_harmless \
    --use_mistral_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless \
    --model_names ${model_name}_with_mistral
#"""

#echo """
# alpaca eval
python generate.py \
    --use_sampling --n_samples 1 --use_alpaca \
    --use_default_prompt --pretrained_model_path ${model}

python generate.py \
    --use_sampling --n_samples 1 --use_alpaca \
    --pretrained_model_path ${model}

python generate.py \
    --use_sampling --n_samples 1 --use_alpaca \
    --use_short_prompt --pretrained_model_path ${model}

python generate.py \
    --use_sampling --n_samples 1 --use_alpaca \
    --use_mistral_prompt --pretrained_model_path ${model}
#"""

#echo """
# testset, only for harmless
python generate.py \
    --use_sampling --n_samples 25 --use_harmless --use_testset \
    --use_default_prompt --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless --use_testset \
    --model_names ${model}_with_default

python generate.py \
    --use_sampling --n_samples 25 --use_harmless --use_testset \
    --pretrained_model_path ${model}

python evaluate.py \
    --config sampling --use_harmless --use_testset \
    --model_names ${model}
#"""
