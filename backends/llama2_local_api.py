"""Llama2 backend API - requires https://github.com/facebookresearch/llama and model weights/tokenizer in directory
/llama2/"""

from typing import List, Dict, Tuple, Any, Optional

from llama import Llama, Dialog
import backends

logger = backends.get_logger(__name__)

MODEL_LLAMA2_7B = "llama-2-7b"
MODEL_LLAMA2_13B = "llama-2-13b"
MODEL_LLAMA2_70B = "llama-2-70b"
MODEL_LLAMA2_7B_C = "llama-2-7b-chat"
MODEL_LLAMA2_13B_C = "llama-2-13b-chat"
MODEL_LLAMA2_70B_C = "llama-2-70b-chat"

SUPPORTED_MODELS = [MODEL_LLAMA2_7B, MODEL_LLAMA2_13B, MODEL_LLAMA2_70B,
                    MODEL_LLAMA2_7B_C, MODEL_LLAMA2_13B_C, MODEL_LLAMA2_70B_C]

NAME = "llama2"


class Llama2Local(backends.Backend):
    """
    Code reference:
    https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py
    https://github.com/facebookresearch/llama/blob/main/example_text_completion.py
    """
    def __init__(self):
        self.chat_models: List = ["llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat"]
        self.temperature: float = -1.
        self.model_loaded: bool = False

    def load_model(self, model_name: str, max_seq_len: int = 512, max_batch_size: int = 8):
        logger.info(f'Start loading llama2 model: {model_name}')

        CKPT_DIR = f'llama2/{model_name}/'
        TOKENIZER_PATH = 'llama2/tokenizer.model'

        self.model = Llama.build(
            ckpt_dir=CKPT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size
        )

        self.model_name = model_name
        self.model_loaded = True

    def generate_response(self, messages: List[Dict], model: str,
                          max_gen_len: Optional[int] = 100, top_p: float = 0.9) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name, chat models for chat-completion, otherwise text completion
        :param max_gen_len: Maximum generation length.
        :param top_p: Top-P sampling parameter.
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading llama2 model: {model}")

        if model in self.chat_models:  # chat completion
            prompt = messages

            dialogs: List[Dialog] = [messages]  # equivalent to chat completion example
            # convert messages to llama Dialog? not done in chat completion example, though

            results = self.model.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=self.temperature,
                top_p=top_p,
            )

            message = results[0]['generation']

            if message["role"] != "assistant":  # safety check
                raise AttributeError("Response message role is " + message["role"] + " but should be 'assistant'")

            response_text = message['content'].strip()
            response = message

        else:  # default (text completion)
            prompt = "\n".join([message["content"] for message in messages])

            prompts = [prompt]  # equivalent to text completion example

            results = self.model.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=self.temperature,
                top_p=top_p,
            )

            generated_text = results[0]['generation']

            response_text = generated_text.strip()

            response = {'response': generated_text}

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
