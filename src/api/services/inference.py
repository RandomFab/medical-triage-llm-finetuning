import uuid
import yaml
from transformers import AutoTokenizer
from config.paths import PARAMS_PATH

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
        
        # Charger le tokenizer et le system prompt pour s'assurer du bon formatage ChatML/Qwen
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        with open(PARAMS_PATH, encoding="utf-8") as f:
            params = yaml.safe_load(f)
        self.system_prompt = params["sft_model"]["system_prompt"]

    def _format_prompt(self, prompt: str) -> str:
        """Formate le prompt avec le Chat Template Qwen incluant le generation prompt."""
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        # Application du formatage correct
        formatted_prompt = self._format_prompt(prompt)
        
        # Paramètres d'échantillonnage de vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            stop_token_ids=self.stop_token_ids, # pour s'assurer que la génération s'arrête aux bons tokens de fin
            repetition_penalty=1.1, # évite les boucle de réponses
        )
        # Nécessite un identifiant unique par requête
        request_id = str(uuid.uuid4())
        
        # Génération asynchrone avec le prompt formaté
        results_generator = self.engine.generate(formatted_prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        generated_text = final_output.outputs[0].text
 
        generated_text = generated_text.replace("<|im_end|>", "").strip() # Nettoyage des tokens de fin
            
        # On retourne le texte généré par la première séquence
        return generated_text