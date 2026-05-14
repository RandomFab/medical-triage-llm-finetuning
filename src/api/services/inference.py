import uuid
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

class VLLMEngine:
    def __init__(self, model_path: str):
        # Configuration des arguments du moteur
        engine_args = AsyncEngineArgs(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1, # Ajuster selon le nombre de GPUs disponibles
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        # Paramètres d'échantillonnage de vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )
        # Nécessite un identifiant unique par requête
        request_id = str(uuid.uuid4())
        
        # Génération asynchrone
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        # On retourne le texte généré par la première séquence
        return final_output.outputs[0].text