from typing import List, Dict, Tuple, Any
from retry import retry
import anthropic
import backends

logger = backends.get_logger(__name__)

MODEL_CLAUDE_13 = "claude-v1.3"
MODEL_CLAUDE_13_100K = "claude-v1.3-100k"
SUPPORTED_MODELS = [MODEL_CLAUDE_13, MODEL_CLAUDE_13_100K]

NAME = "anthropic"


class Anthropic(backends.Backend):
    def __init__(self):
        creds = backends.load_credentials(NAME)
        self.client = anthropic.Client(creds[NAME]["api_key"])
        self.temperature: float = -1.

    @retry(tries=3, delay=0, logger=logger)
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[str, Any, str]:
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
        prompt = ''
        for message in messages:
            if message['role'] == 'assistant':
                prompt += f'{anthropic.AI_PROMPT} {message["content"]}'
            elif message['role'] == 'user':
                prompt += f'{anthropic.HUMAN_PROMPT} {message["content"]}'

        prompt += anthropic.AI_PROMPT

        response = self.client.completion(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT, '\n'],
            model=model,
            temperature=self.temperature,
            max_tokens_to_sample=100
        )
        response_text = response['completion'].strip()
        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
