import os
import copy
import numpy as np
from typing import List, Tuple
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm import LLM, SamplingParams

class LLMGenerator(object):
    def __init__(self, model_path, cuda_device="0"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        self.model = LLM(
            model=model_path,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enable_prefix_caching=True,
            tensor_parallel_size=len(cuda_device.split(","))
        )
        self.params = SamplingParams(
            temperature=1.0,
            top_p=0.8,
            max_tokens=1024,
            seed=0
        )

    def run(
            self,
            messages: List[ChatCompletionMessageParam] | List[List[ChatCompletionMessageParam]],
            sampling_params: SamplingParams = None,
        ):
        if not sampling_params:
            sampling_params = self.params
        return self.model.chat(
            messages,
            sampling_params=sampling_params,
            add_generation_prompt=False,
            continue_final_message=True
        )


class LLMClassifier(object):
    def __init__(
            self,
            model_path: str,
            gpu_device: str = "0",
        ):
        self.generator = LLMGenerator(model_path, gpu_device)
        self.choice_ids = {}

    def _set_choices(self, choices: List[str]) -> None:
        tokenizer = self.generator.model.get_tokenizer()
        self.choice_ids = {tokenizer.encode(choice, add_special_tokens=False)[0]: choice for choice in choices}
        assert len(self.choice_ids) == len(choices), "Choices must be unique"
        self.prob_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,
            logprobs=len(self.choice_ids),
            allowed_token_ids=list(self.choice_ids.keys())
        )

    def run(
        self,
        messages: List[ChatCompletionMessageParam] | List[List[ChatCompletionMessageParam]],
        choices: List[str] = None,
        sampling_params: SamplingParams = None,
    ) -> List[dict]:
        if self.choice_ids == {} and choices is None:
            raise ValueError("Choices must be set before calling get_choices_probs")
        if choices:
            self._set_choices(choices)
        if not sampling_params:
            sampling_params = self.prob_params
        responses = self.generator.run(messages, sampling_params=sampling_params)
        probs = []
        for response in responses:
            logprobs = response.outputs[0].logprobs[0]
            probs.append({self.choice_ids[token_id]: np.exp(logprob.logprob) for token_id, logprob in logprobs.items() if token_id in self.choice_ids})
        return probs

    def run_multiturn(
        self,
        messages: List[ChatCompletionMessageParam] | List[List[ChatCompletionMessageParam]],
        choices: List[str] = None,
        stop_turn: List[str] = None,
        gen_samp_params: SamplingParams = None,
        prob_samp_params: SamplingParams = None,
        repeat_times: int = 1
    ) -> Tuple[List[dict], List[str]]:

        for stop in stop_turn:
            sampling_params = gen_samp_params if gen_samp_params else self.generator.params
            sampling_params.stop = [stop]
            sampling_params.n = repeat_times
            responses = self.generator.run(messages, sampling_params=sampling_params)
            new_messages = []
            for message, response in zip(messages, responses):
                for i in range(repeat_times):
                    message_copy = copy.deepcopy(message)
                    message_copy[-1]["content"] += response.outputs[i].text + stop
                    new_messages.append(message_copy)
            probs = self.run(new_messages, choices, prob_samp_params)
        # Reshape all_probs and all_messages into (num_samples, repeat_times) format
        new_messages = [new_messages[i*repeat_times:(i+1)*repeat_times] for i in range(len(messages))]
        probs = [probs[i*repeat_times:(i+1)*repeat_times] for i in range(len(messages))]
        return probs, new_messages
        