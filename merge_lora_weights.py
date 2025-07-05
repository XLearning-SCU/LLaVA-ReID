import argparse
import os

import torch
from accelerate import init_empty_weights
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoTokenizer

from model.llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM


def merge_question_model(model_path, save_model_path):
    print("Load question model from {}".format(model_path))
    config = LlavaQwenConfig.from_pretrained(model_path)
    model = LlavaQwenForCausalLM.from_pretrained(config._name_or_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    # print(model)
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        # print(model)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        # print(model)
        print("Model is merged and loaded...")

    tokenizer = AutoTokenizer.from_pretrained(config._name_or_path, padding_side="left")
    config._name_or_path = save_model_path
    del config.quantization_config
    model.config = config
    model.config.tokenizer_padding_side = 'left'
    return model, tokenizer


def merge_lora(args):
    model, tokenizer = merge_question_model(args.model_path, args.save_model_path)

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
