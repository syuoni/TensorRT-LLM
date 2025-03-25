from typing import List, Tuple
import lm_eval
from lm_eval.api.model import TemplateLM

from tensorrt_llm import LLM, SamplingParams

class LmEvalWrapper(TemplateLM):
    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm

    @property
    def eot_token_id(self) -> int:
        return self.llm.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.llm.tokenizer.encode(string, **kwargs)

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError()

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        outputs = []
        for request in requests:
            prompt, gen_kwargs = request.args
            sampling_params = SamplingParams(
                max_tokens=gen_kwargs.get("max_gen_toks", 256),
                top_p=gen_kwargs.get("top_p", None),
                temperature=gen_kwargs.get("temperature", None),
                stop=gen_kwargs.get("until", None),
                include_stop_str_in_output=False
            )
            output = self.llm.generate_async(prompt, sampling_params=sampling_params)
            outputs.append(output)

        for output in outputs:
            output.result()

        return [output.outputs[0].text for output in outputs]



llm = LLM("/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Meta-Llama-3.1-8B", backend="pytorch")


# suppose you've defined a custom lm_eval.api.Task subclass in your own external codebase
# from my_tasks import MyTask1
# ...

# # create your model (could be running finetuning with some custom modeling code)
# my_model = initialize_my_model()
# ...

# # instantiate an LM subclass that takes your initialized model and can run
# # - `Your_LM.loglikelihood()`
# # - `Your_LM.loglikelihood_rolling()`
# # - `Your_LM.generate_until()`
# lm_obj = Your_LM(model=my_model, batch_size=16)

# optional: the task_manager indexes tasks including ones
# specified by the user through `include_path`.
# task_manager = lm_eval.tasks.TaskManager()

# To get a task dict for `evaluate`
task_dict = lm_eval.tasks.get_task_dict("gsm8k")


results = lm_eval.evaluate(
    lm=LmEvalWrapper(llm),
    task_dict=task_dict,
)
print(results['results'])
breakpoint()
