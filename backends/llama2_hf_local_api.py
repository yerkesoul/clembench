"""Llama2 backend API - uses HuggingFace transformers to load weights and run inference"""

from typing import List, Dict, Tuple, Any, Optional
import torch
import backends
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

logger = backends.get_logger(__name__)

MODEL_LLAMA2_7B_HF = "llama-2-7b-hf"
MODEL_LLAMA2_13B_HF = "llama-2-13b-hf"
MODEL_LLAMA2_70B_HF = "llama-2-70b-hf"
MODEL_LLAMA2_7B_C_HF = "llama-2-7b-chat-hf"
MODEL_LLAMA2_13B_C_HF = "llama-2-13b-chat-hf"
MODEL_LLAMA2_70B_C_HF = "llama-2-70b-chat-hf"

SUPPORTED_MODELS = [MODEL_LLAMA2_7B_HF, MODEL_LLAMA2_13B_HF, MODEL_LLAMA2_70B_HF,
                    MODEL_LLAMA2_7B_C_HF, MODEL_LLAMA2_13B_C_HF, MODEL_LLAMA2_70B_C_HF]

NAME = "llama2-hf"


class Llama2LocalHF(backends.Backend):
    def __init__(self):
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        self.api_key = creds["huggingface"]["api_key"]

        self.chat_models: List = [MODEL_LLAMA2_7B_C_HF, MODEL_LLAMA2_13B_C_HF, MODEL_LLAMA2_70B_C_HF]
        self.temperature: float = -1.
        self.model_loaded: bool = False

    def load_model(self, model_name: str, max_seq_len: int = 512, max_batch_size: int = 8):
        assert model_name in SUPPORTED_MODELS, f"{model_name} is not supported, please make sure the model name is correct."
        logger.info(f'Start loading llama2-hf model: {model_name}')
        # full HF model id string:
        hf_id_str = f"meta-llama/{model_name.capitalize()}"
        # load tokenizer and model:
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id_str, token=self.api_key, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(hf_id_str, token=self.api_key, device_map="auto")
        # use CUDA if available:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # init HF pipeline:
        self.model_pipeline = pipeline('text-generation', tokenizer=self.tokenizer, model=self.model, device_map="auto")

        self.model_name = model_name
        self.model_loaded = True

    def generate_response(self, messages: List[Dict], model: str,
                          max_new_tokens: Optional[int] = 100, top_p: float = 0.9) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name, chat models for chat-completion, otherwise text completion
        :param max_new_tokens: Maximum generation length.
        :param top_p: Top-P sampling parameter.
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading llama2-hf model: {model}")

        # greedy decoding:
        do_sample: bool = False
        if self.temperature > 0.0:
            do_sample = True

        if model in self.chat_models:  # chat completion

            # apply chat template & tokenize
            prompt_tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            prompt = {"inputs": prompt_text, "max_new_tokens": max_new_tokens,
                      "temperature": self.temperature}

            model_output_ids = self.model.generate(
                prompt_tokens,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=top_p
            )

            model_output = self.tokenizer.decode(model_output_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)

            response = {
                "role": "assistant",
                "content": model_output,
            }

            response_text = model_output.replace(prompt_text, '').strip()

        else:  # default (text completion)
            prompt = "\n".join([message["content"] for message in messages])

            prompt_tokens = self.tokenizer.encode(
                prompt,
                add_bos_token=True,
                add_eos_token=False,
            )

            model_output_ids = self.model.generate(
                prompt_tokens,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=top_p
            )

            model_output = self.tokenizer.decode(model_output_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)

            response_text = model_output.replace(prompt, '').strip()

            response = {'response': response_text}

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
