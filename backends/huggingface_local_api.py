from typing import List, Dict, Tuple, Any
import torch
import anthropic
import backends
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, pipeline,  AutoModelForCausalLM, GPTNeoXForCausalLM

logger = backends.get_logger(__name__)

MODEL_GOOGLE_FLAN_T5 = "flan-t5-xxl"
MODEL_VICUNA_13B = "vicuna-13b"
MODEL_OPEN_ASSISTANT = "oasst-12b"
MODEL_KOALA_13B = "koala-13b"
MODEL_FALCON_7B = "falcon-7b"
MODEL_FALCON_40B = "falcon-40b"
SUPPORTED_MODELS = [MODEL_GOOGLE_FLAN_T5, MODEL_VICUNA_13B, MODEL_KOALA_13B, MODEL_OPEN_ASSISTANT, MODEL_FALCON_7B, MODEL_FALCON_40B]

NAME = "huggingface"


class HuggingfaceLocal(backends.Backend):
    def __init__(self):
        self.temperature: float = -1.
        self.model_loaded = False

    def load_model(self, model_name):

        logger.info(f'Start loading huggingface model: {model_name}')

        CACHE_DIR = 'huggingface_cache'

        if 'flan' in model_name:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl", device_map="auto", cache_dir=CACHE_DIR)
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto",
                                                                    cache_dir=CACHE_DIR)
        elif 'falcon' in model_name:
            model_cpt = 'tiiuae/falcon-7b-instruct' if '7b' in model_name else 'tiiuae/falcon-40b-instruct'
            self.tokenizer = AutoTokenizer.from_pretrained(model_cpt, device_map="auto",
                                                           cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_cpt, device_map="auto", load_in_8bit = True,  low_cpu_mem_usage = True, torch_dtype=torch.float16, trust_remote_code=True, cache_dir=CACHE_DIR
            )

        elif 'oasst' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
                                                           device_map="auto", cache_dir=CACHE_DIR)
            self.model = GPTNeoXForCausalLM.from_pretrained(
                "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
                device_map="auto", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR)
        elif 'koala' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/koala-13B-HF", device_map="auto",
                                                           cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/koala-13B-HF", device_map="auto", cache_dir=CACHE_DIR
            )
        elif 'vicuna' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/Wizard-Vicuna-13B-Uncensored-HF",
                                                           device_map="auto", cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Wizard-Vicuna-13B-Uncensored-HF", device_map="auto", cache_dir=CACHE_DIR
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_pipeline = pipeline('text-generation', tokenizer=self.tokenizer, model=self.model, device_map="auto")
        self.model_name = model_name
        self.model_loaded = True

    def generate_response(self, messages: List[Dict], model: str) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading huggingface model: {model}")

        prompt_text = ''

        if 'oasst' in self.model_name:
            for message in messages:
                content = message["content"]
                if message['role'] == 'assistant':
                    prompt_text += '<|assistant|>' + content + '<|endoftext|>'
                elif message['role'] == 'user':
                    prompt_text += '<|prompter|>' + content + '<|endoftext|>'

            prompt_text += '<|assistant|>'

        elif 'falcon' in self.model_name:
            for message in messages:
                content = message["content"]
                if message['role'] == 'assistant':
                    prompt_text += content + '<|endoftext|>'
                elif message['role'] == 'user':
                    prompt_text += content + '<|endoftext|>'

        else:
            for message in messages:
                content = message["content"]
                if message['role'] == 'assistant':
                    prompt_text += f'{anthropic.AI_PROMPT} {content}.'
                elif message['role'] == 'user':
                    prompt_text += f'{anthropic.HUMAN_PROMPT} {content}'

            prompt_text += anthropic.AI_PROMPT

        if self.temperature == 0:
            self.temperature += 0.01

        prompt = {"inputs": prompt_text, "max_new_tokens": 100,
                  "temperature": self.temperature, "return_full_text": False}

        if self.temperature == 0:
            self.temperature += 0.01

        pipeline_output = self.model_pipeline(f"{prompt_text}",
                            temperature=self.temperature,
                            max_new_tokens=100,
                            return_full_text=False,
                            do_sample=True)

        generated_text = pipeline_output[0]['generated_text']

        response_text = generated_text.replace(prompt_text, '')\
            .replace('<pad>', '')\
            .replace('<s>', '')\
            .replace('</s>','')\
            .replace('<|endoftext|>', '')\
            .replace('<|assistant|>','').strip()

        response = {'response': generated_text}
        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
