import os
import torch
import numpy as np
import asyncio
import clip
import regex as re
import template
import time
import warnings
import matplotlib.pyplot as plt
from mydataset import Dataset, subsampled_data
from torch.utils.data import DataLoader
from log import TrainingLogger
from configs import parse_args, set_all_seed
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from torch import autocast, nn
from diffusers import StableDiffusionPipeline, DDIMScheduler,DiffusionPipeline
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, logging
from peft import LoraConfig, get_peft_model, PeftModel
from save_utils import save_batch_results, save_result
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
# --------------------  --------------------
def compute_f1(generated, reference):
    """ F1-score """
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()
    common = set(gen_tokens) & set(ref_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

# --------------------  --------------------
class SoftPromptOptimizer:
    """
    
    """
    def __init__(self, whitebox_model, blackbox_model, evaluator, args):
        """
        
        
        
        """
        self.whitebox = whitebox_model
        self.blackbox = blackbox_model
        self.evaluator = evaluator
        self.args = args
        
        
        # 
        self.embed_dim = self.whitebox.embedding.shape[1]
        
        # 
        self.z_t = torch.zeros(args.intrinsic_dim).to(args.cuda)
        
        
        
        # 
        self.mu = getattr(args, "mu", 0.1)  # 
        self.lr = getattr(args, "soft_lr", 0.1)  # 
        self.n_prompt_tokens = getattr(args, "n_prompt_tokens", 5)  # 
        # 
        self.A = self._initialize_projection_matrix()
        self.n_directions = getattr(args,'soft_n_directions',5)
    
    def _initialize_projection_matrix(self):
        """"""
        A = torch.nn.Linear(
            self.args.intrinsic_dim, 
            self.n_prompt_tokens * self.embed_dim, 
            bias=False
        ).to(self.args.cuda)
        
        # 
        random_proj = getattr(self.args, "random_proj", "uniform")
        
        if random_proj == "normal":
            # 
            mu_hat = self.whitebox.embedding.mean().item()
            std_hat = self.whitebox.embedding.std().item()
            
            # 
            alpha = getattr(self.args, "alpha", 1.0)
            sigma = getattr(self.args, "sigma", 1.0)
            
            mu = 0.0
            std = alpha * std_hat / (np.sqrt(self.args.intrinsic_dim) * sigma)
            
            print(f"[Embedding] mu: {mu_hat}, std: {std_hat} [RandProj] mu: {mu}, std: {std}")
            torch.nn.init.normal_(A.weight, mean=mu, std=std)
            
        elif random_proj == "uniform":
            # 
            torch.nn.init.uniform_(A.weight, -1, 1)
        
        else:
            raise ValueError(f"Unknown random_proj type: {random_proj}")
        
        print(f"A weight mean: {A.weight.mean().item()}, std: {A.weight.std().item()}")
        return A
    
    def get_soft_prompt_embeds(self):
        """"""
        # 
        z_projected = self.A(self.z_t.unsqueeze(0))  # [1, n_prompt_tokens * embed_dim]
        
        #  [1, n_prompt_tokens, embed_dim] 
        embeds = z_projected.view(1, self.n_prompt_tokens, self.embed_dim)
        
        #  ()
        embeds = embeds.to(self.whitebox.model.dtype)

        #  [1, n_prompt_tokens, embed_dim] 
        return embeds
    
    async def optimize_step(self, prompts, batch_idx=None, epoch=None, output_dir=None):
        """"""
        z_t_tmp = self.z_t.unsqueeze(0)  # [1, intrinsic_dim]
        total_grad = torch.zeros_like(self.z_t)  # Initialize accumulator
        
        for _ in range(self.n_directions):

            # 
            noise = torch.normal(mean=0.0, std=1.0, size=z_t_tmp.shape).to(self.args.device)
            # 
            z_t_pos = z_t_tmp + self.mu * noise
            z_t_neg = z_t_tmp - self.mu * noise
            
            # 
            Az_pos = self.A(z_t_pos).view(1, self.n_prompt_tokens, self.embed_dim)
            Az_neg = self.A(z_t_neg).view(1, self.n_prompt_tokens, self.embed_dim)

            # 
            Az_pos = Az_pos.to(self.whitebox.model.dtype)
            Az_neg = Az_neg.to(self.whitebox.model.dtype)

            # 
            pos_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=Az_pos)

            # 
            neg_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=Az_neg)
            
            # 
            pos_results = await self.blackbox.generate(pos_prompts)
            neg_results = await self.blackbox.generate(neg_prompts)
            
            # 
            if self.evaluator.eval_type == "image":
                pos_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(pos_prompts, pos_results)]
                neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
            else:
                pos_scores = [self.evaluator.evaluate(r) for r in pos_results]
                neg_scores = [self.evaluator.evaluate(r) for r in neg_results]
            
            # # 
            # pos_score_avg = np.mean(pos_scores)
            # neg_score_avg = np.mean(neg_scores)

        
            # print(f"{_}: Pos Score: {pos_score_avg:.4f}, Neg Score: {neg_score_avg:.4f}")
            
            # # 
            # score_diff = pos_score_avg - neg_score_avg
            #g_t_hat = ((score_diff / (2 * self.mu)) * noise).squeeze(0)
            # 
            #self.z_t = self.z_t + self.lr * g_t_hat
            score_diff = np.mean(pos_scores) - np.mean(neg_scores)
            
            # 
            total_grad += (score_diff / (2 * self.mu)) * noise.squeeze(0)
        
        # 
        avg_grad = total_grad / self.n_directions
        self.z_t += self.lr * avg_grad  # 
        
        # 
        current_soft_prompt = self.get_soft_prompt_embeds()
        current_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=current_soft_prompt)
        current_results = await self.blackbox.generate(current_prompts)
        
        # 
        if output_dir is not None and batch_idx is not None and epoch is not None and self.args.gene_image == 'True':
            save_batch_results(
                current_results,
                current_prompts,
                output_dir,
                epoch=epoch,
                method="soft_prompt",
                batch_idx=batch_idx
            )
        
        # 
        if self.evaluator.eval_type == "image":
            current_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(current_prompts, current_results)]
        else:
            current_scores = [self.evaluator.evaluate(r) for r in current_results]
        
        current_score_avg = np.mean(current_scores)
        print(f"Current Score: {current_score_avg:.4f}")
        
        return {
            "score": current_score_avg,
            "soft_prompt": self.z_t.cpu().detach().clone(),
            "prompts": current_prompts,
            "results": current_results
        }
    
    async def optimize(self, prompts, epochs, batch_idx=None, output_dir=None):
        """"""
        best_score = -float('inf')
        best_z_t = None
        scores_history = []
        
        for epoch in range(epochs):
            print(f"Soft Prompt Optimization Epoch {epoch+1}/{epochs}")
            result = await self.optimize_step(
                prompts, 
                batch_idx=batch_idx, 
                epoch=epoch,
                output_dir=output_dir
            )
            scores_history.append(result["score"])
            
            # 
            if result["score"] > best_score:
                best_score = result["score"]
                best_z_t = self.z_t.cpu().detach().clone()
            
            print(f"Epoch {epoch+1}, Score: {result['score']:.4f}, Best Score: {best_score:.4f}")
        
        # 
        if best_z_t is not None:
            self.z_t = best_z_t.to(self.args.device)
        
        return {
            "best_score": best_score,
            "best_soft_prompt": self.z_t,
            "scores_history": scores_history,
            "final_soft_prompt_embeds": self.get_soft_prompt_embeds()
        }
    
    def save(self, path):
        """"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            "z_t": self.z_t,
            "A_state_dict": self.A.state_dict(),
            "n_prompt_tokens": self.n_prompt_tokens,
            "embed_dim": self.embed_dim,
            "intrinsic_dim": self.args.intrinsic_dim
        }, os.path.join(path, "soft_prompt.pt"))
        print(f"Saved soft prompt to {path}")
    
    @classmethod
    def load(cls, path, whitebox_model, blackbox_model, evaluator, args):
        """"""
        checkpoint = torch.load(os.path.join(path, "soft_prompt.pt"), map_location=whitebox_model.device)
        
        # 
        args.n_prompt_tokens = checkpoint["n_prompt_tokens"]
        args.intrinsic_dim = checkpoint["intrinsic_dim"]
        
        # 
        optimizer = cls(whitebox_model, blackbox_model, evaluator, args)
        
        # 
        optimizer.z_t = checkpoint["z_t"].to(whitebox_model.device)
        optimizer.A.load_state_dict(checkpoint["A_state_dict"])
        
        return optimizer

class TextEvaluator:
    #
    """ F1-score"""
    def __init__(self, reference):
        self.reference = reference

    def evaluate(self, generated_text):
        return compute_f1(generated_text, self.reference)

class AestheticMlp(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class ImageEvaluator:
    """
     Aesthetic、CLIPScore、PickScore 
    """
    def __init__(self, cache_dir, device, args):
        self.device = device
        #  CLIP 
        clip_cache_dir = os.path.join(cache_dir, "clip_model")
        # print(f"clip_cache_dir:{clip_cache_dir}")
        os.makedirs(clip_cache_dir, exist_ok=True)
        model_path = os.path.join(clip_cache_dir, "ViT-L-14.pt")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device, download_root=clip_cache_dir)
        
        #  Aesthetic 
        self.aes_model = AestheticMlp(768)
        state_dict = torch.load("./cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth", map_location=device)
        self.aes_model.load_state_dict(state_dict)
        self.aes_model.to(device)
        self.aes_model.eval()

        #  PickScore 
        pickscore_cache_dir = os.path.join(cache_dir, "pickscore_model")
        os.makedirs(pickscore_cache_dir, exist_ok=True)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path_pick = "yuvalkirstain/PickScore_v1"
        self.pickscore_processor = AutoProcessor.from_pretrained(processor_path, cache_dir=pickscore_cache_dir, device=device)
        self.pickscore_model = AutoModel.from_pretrained(model_path_pick, cache_dir=pickscore_cache_dir).eval().to(device)

        # 
        self.lambda_aes = args.lambda_aes
        self.lambda_clip = args.lambda_clip
        self.lambda_pick = args.lambda_pick
    
    def get_clip_features(self, image, is_batched=False):
        if not is_batched:
            image = self.clip_preprocess(image).unsqueeze(0)
            image = image.to(self.device)
        else:
            images = [self.clip_preprocess(i) for i in image]
            image = torch.stack(images).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features

    def get_clip_score(self, image_features, prompt):
        tokens = clip.tokenize(prompt[:77], truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens[:77])
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            score = (image_features @ text_features.t()).item()
        return score

    def get_aesthetic_score(self, image_features):
        features = image_features.cpu().detach().numpy()
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        normalized = features / norm
        tensor_features = torch.tensor(normalized, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            prediction = self.aes_model(tensor_features)
        return prediction.item()

    def get_pick_score(self, prompt, image):
        image_inputs = self.pickscore_processor(images=image, return_tensors="pt").to(self.device)
        text_inputs = self.pickscore_processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embs = self.pickscore_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = self.pickscore_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            score = (self.pickscore_model.logit_scale.exp() * (text_embs * image_embs).sum()).item()
        return score

    def evaluate(self, prompt, image):
        features = self.get_clip_features(image)
        aes_score = self.get_aesthetic_score(features)
        clip_score = self.get_clip_score(features, prompt)
        pick_score = self.get_pick_score(prompt, image)
        final_score = (
            aes_score * self.lambda_aes + 
            clip_score * self.lambda_clip + 
            pick_score * self.lambda_pick
        )
        # 返回详细分数
        return {
            'total': final_score,
            'aesthetic': aes_score,
            'clip': clip_score,
            'pick': pick_score
        }

class Evaluator:
    """
      - eval_type=="text": TextEvaluator
      - eval_type=="image": ImageEvaluator
    """
    def __init__(self, eval_type, **kwargs):
        self.eval_type = eval_type
        if eval_type == "text":
            self.evaluator = TextEvaluator(**kwargs)
        elif eval_type == "image":
            self.evaluator = ImageEvaluator(**kwargs)
        else:
            raise ValueError("Unknown evaluation type")

    def evaluate(self, prompt, result):
        if self.eval_type == "image":
            return self.evaluator.evaluate(prompt, result)
        else:  # text evaluation
            return self.evaluator.evaluate(result)

# --------------------    --------------------
class WhiteBoxModel:
    MODEL_TARGET_MAP = {
        "llama": ["q_proj", "v_proj"],
        "vicuna": ["q_proj", "v_proj"],
        "gpt2": ["c_attn"],
        "bert": ["query", "value"],
        "t5": ["q", "v"],
        "default": ["k_proj", "v_proj"]
    }

    MODEL_LAYER_PATTERNS = {
        "llama": r'\.layers\.(\d+)',
        "vicuna": r'\.layers\.(\d+)',
        "gpt2": r'h\.(\d+)',
        "bert": r'encoder\.layer\.(\d+)',
        "t5": r'block\.(\d+)',
        "default": r'\.layers\.(\d+)'
    }

    def __init__(self, model_name: str, hf_token: str = None, n_last_layers: int = 1, device: str = None, lora_rank: int = 4):
        self.model_type = self._detect_model_type(model_name)
        self.model, self.tokenizer = self._load_model(model_name, hf_token, device)
        self._inject_lora(n_last_layers=n_last_layers, lora_rank=lora_rank)
        self.embedding = self.model.get_input_embeddings().weight.clone().to(device)

    def load_lora(self, lora_dir: str):
        """"""
        # 
        required_files = ["adapter_model.safetensors", "adapter_config.json"]
        for f in required_files:
            if not os.path.exists(os.path.join(lora_dir, f)):
                print(f"os.path.exists(os.path.join(lora_dir, f)):{os.path.join(lora_dir, f)}")
                raise FileNotFoundError(f"Missing LoRA file: {f}")
     
        # 
        self.model.load_adapter(lora_dir, adapter_name="adapter_model")
        self.model.set_adapter("adapter_model")
        print(f": {lora_dir}")

    def merge_lora(self):
        """"""
        original_weight = self.model.transformer.h[-1].attn.c_attn.weight.clone()
        self.model.merge_and_unload()
        merged_weight = self.model.transformer.h[-1].attn.c_attn.weight
        if torch.allclose(original_weight, merged_weight, atol=1e-5):
            raise RuntimeError("LoRA failed!")
        print("LoRA succeed")

    def _detect_model_type(self, model_name: str) -> str:
        model_name = model_name.lower()
        if "llama" in model_name:
            return "llama"
        elif "vicuna" in model_name:
            return "vicuna"
        elif "gpt2" in model_name:
            return "gpt2"
        elif "promtist" in model_name:
            return "gpt2"
        elif "sft" in model_name:
            return "gpt2"
        elif "bert" in model_name:
            return "bert"
        elif "t5" in model_name:
            return "t5"
        return "default"


    def _load_model(self, model_name: str, hf_token: str, device: str = None):
        if "vicuna" in model_name.lower():
            return self._load_vicuna_model(model_name, hf_token, device)
        
        if "promtist" in model_name.lower():
            return self.load_promtist(model_name, hf_token, device)
        
        if "sft" in model_name.lower():
            return self.load_sft(model_name, hf_token, device)
        
        if hf_token:
            from huggingface_hub import login, HfFolder
            login(token=hf_token)
            HfFolder.save_token(hf_token)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map= device,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"./cache/{model_name}")
        tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token
        return model, tokenizer
    def load_promtist(self, model_name: str, hf_token: str = None, device: str = None):
        print(f"load_promtist model，use gpt2 architecture")
        prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist", device_map = device, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=f"./cache/{model_name}")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return prompter_model, tokenizer
    
    def load_sft(self, model_name: str, hf_token: str = None, device: str = None):
        print("Loading SFT model...")
        state_dict_path = "./cache/sft/sft_gpt.bin"
        
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(target_device)

        def load_sft_model(state_dict_path, base_model_name="gpt2", device=torch_device):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16
            ).to(device)
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model

        sft_model = load_sft_model(state_dict_path, device=torch_device)
        
        sft_model.to(torch_device)
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        param_device = next(sft_model.parameters()).device
        print(f"model parameters have been loaded to : {param_device}")
        
        print("SFT model loaded successfully.")
        return sft_model, tokenizer
    
    def _load_vicuna_model(self, model_name: str, hf_token: str = None, device: str = None):
        """
          "lmsys/vicuna-13b-v1.3"
        """
        from huggingface_hub import snapshot_download, HfApi

        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        try:
            if not os.path.exists(model_name):

                if not model_name.startswith("lmsys/"):
                    repo_id = f"lmsys/{model_name}"
                else:
                    repo_id = model_name
                
                try:
                    HfApi().model_info(repo_id, token=hf_token)
                except Exception as e:
                    raise ValueError(
                        f"can't visit model {repo_id}"
                    ) from e

                model_path = snapshot_download(
                    repo_id=repo_id,
                    token=hf_token,
                    ignore_patterns=["*.safetensors"],  
                    resume_download=True,
                    local_dir = f"./cache/{repo_id}"
                )
            else:
                model_path = model_name

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                token=hf_token,
                model_max_length=1024,
                padding_side="left"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map= device,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                token=hf_token
            )

            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            
            return model, tokenizer

        except Exception as e:
            raise RuntimeError(
                f"loading Vicuna failed:{str(e)}"
            ) from e
    
    def _inject_lora(self, lora_rank: int = 4, custom_targets: list = None, n_last_layers: int = 1):
        base_targets = custom_targets or self.MODEL_TARGET_MAP.get(self.model_type, self.MODEL_TARGET_MAP["default"])
        
        # 
        candidate_modules = []
        for name, _ in self.model.named_modules():
            if any(target in name for target in base_targets):
                candidate_modules.append(name)
        
        # 
        layer_pattern = self.MODEL_LAYER_PATTERNS[self.model_type]
        layer_re = re.compile(layer_pattern)
        
        layer_info = []
        for name in candidate_modules:
            match = layer_re.search(name)
            if match:
                layer_num = int(match.group(1))
                layer_info.append((layer_num, name))
        
        if not layer_info:
            raise ValueError("can't find the required Module")
        
        max_layer = max(layer_num for layer_num, _ in layer_info)
        selected_layers = range(max_layer - n_last_layers + 1, max_layer + 1)
        target_modules = [name for layer_num, name in layer_info if layer_num in selected_layers]
        
        # 
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 
        for name, param in self.model.named_parameters():
            # if 'lora_A' in name or 'lora_B' in name:
            if 'lora_B' in name:
                nn.init.zeros_(param)
        
        self.model.print_trainable_parameters()
        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                print(f"Parameter {name} set_flat_params: {param.data}")

    def get_flat_params(self) -> torch.Tensor:
        lora =  torch.cat([p.detach().flatten() for p in self.model.parameters() if p.requires_grad])
        # print(f"get_flat_lora :{lora} ")
        return lora

    def set_flat_params(self, flat_params: torch.Tensor):
        ptr = 0
        for p in self.model.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.data.copy_(flat_params[ptr:ptr+numel].reshape(p.shape))
                ptr += numel

    def generate(self, prompts: list, max_new_tokens: int = 75, soft_prompt_embeds=None) -> list:
        """
        
        """
        start_time = time.time()

        print(f"white box input：\n{prompts}")
        
        if self.model_type == "gpt2":
            new_prompts = [p + " Rephrase:" for p in prompts]
            
            input_ids = self.tokenizer(
                new_prompts, 
                return_tensors="pt", 
                padding=True
            ).input_ids.to(self.model.device)
            
            if soft_prompt_embeds is not None:
                # 
                input_embeds = self.model.get_input_embeddings()(input_ids)
                
                # 
                batch_size = input_embeds.shape[0]
                if soft_prompt_embeds.shape[0] == 1 and batch_size > 1:
                    soft_prompt_embeds = soft_prompt_embeds.repeat(batch_size, 1, 1)
                
                # : [BOS, soft_prompt, input_text]
                input_embeds_with_soft = torch.cat([
                    input_embeds[:, :1, :],  #  (BOS)
                    soft_prompt_embeds,      # 
                    input_embeds[:, 1:, :]   # 
                ], dim=1)
                
                #  attention_mask
                attention_mask = torch.ones(
                    (batch_size, input_embeds_with_soft.shape[1]),
                    device=input_embeds.device
                )
                
                # 
                outputs = self.model.generate(
                    inputs_embeds=input_embeds_with_soft,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=8,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    length_penalty=-1.0
                )
            else:
                #  input_ids 
                outputs = self.model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=8,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    length_penalty=-1.0
                )
            
            # 
            output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            results = []
            for i, text in enumerate(output_texts):
                processed = text.replace(new_prompts[i], "").strip()
                results.append(processed)

            print(f"white box model's outputs after processing：\n{results}")
            print(f"白盒生成耗时: {time.time()-start_time:.2f}s")
            return results 
        
        # 
        input_token = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = input_token.input_ids.to(self.model.device)
        attention_mask = input_token.attention_mask.to(self.model.device)

        if soft_prompt_embeds is not None:
            # 
            input_embed = self.embedding[input_ids]
            
            #  soft_prompt_embeds  batch_size  input_embed 
            batch_size = input_embed.shape[0]
            if soft_prompt_embeds.shape[0] == 1 and batch_size > 1:
                soft_prompt_embeds = soft_prompt_embeds.repeat(batch_size, 1, 1)
            
            # : [BOS, soft_prompt, input_text]
            input_embed_with_soft = torch.cat([
                input_embed[:, :1, :],  #  (BOS)
                soft_prompt_embeds,     # 
                input_embed[:, 1:, :]   # 
            ], dim=1)
            
            #  attention_mask
            soft_prompt_length = soft_prompt_embeds.shape[1]
            soft_attention_mask = torch.ones(
                (batch_size, soft_prompt_length),
                device=attention_mask.device
            )
            attention_mask_with_soft = torch.cat([
                attention_mask[:, :1],    #  mask
                soft_attention_mask,      #  mask
                attention_mask[:, 1:]     #  mask
            ], dim=1)
            
            # 
            outputs = self.model.generate(
                inputs_embeds=input_embed_with_soft,
                attention_mask=attention_mask_with_soft,
                do_sample=False, 
                max_new_tokens=max_new_tokens, 
                num_beams=8, 
                num_return_sequences=1,  
                eos_token_id=self.tokenizer.eos_token_id, 
                pad_token_id=self.tokenizer.eos_token_id, 
                length_penalty=-1.0
            )
        else:
            #  input_embed 
            input_embed = self.embedding[input_ids]
            outputs = self.model.generate(
                inputs_embeds=input_embed,
                attention_mask=attention_mask,
                do_sample=False, 
                max_new_tokens=max_new_tokens, 
                num_beams=8, 
                num_return_sequences=1,  
                eos_token_id=self.tokenizer.eos_token_id, 
                pad_token_id=self.tokenizer.eos_token_id, 
                length_penalty=-1.0
            )
        
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"white box's output：\n{results}")
        return results
# --------------------  --------------------
class BlackBoxModel:
    """
    
      - model_type=="text" gpt2-xl 
      - model_type=="image" Stable Diffusion 1.4 
    """
    def __init__(self, model_type: str = "text", model_name: str = None, device: str = "cuda:0", batch_size: int = 1, max_workers: int = 4):
        self.device = device
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model_type = model_type

        if model_type == "text":
            if model_name is None:
                model_name = "gpt2-xl"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_type == "image":
            if model_name is None:
                model_name = "CompVis/stable-diffusion-v1-4"  # SD 1.4
                # model_name = "sd-legacy/stable-diffusion-v1-5"  # SD 1.4
            if model_name == 'stabilityai/stable-diffusion-xl-base-1.0':
                self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    use_auth_token=True
                ).to(device)
            self.pipe.set_progress_bar_config(disable=True)
        else:
            raise ValueError("Unknown model_type for BlackBoxModel")

    async def generate(self, prompts: list, **kwargs) -> list:
        # start_time = time.time()
        # print(f"{prompts}")
        results = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            try:
                if self.model_type == "text":
                    batch_results = await self._generate_text_batch(batch, kwargs)
                elif self.model_type == "image":
                    # print(f"{batch}")
                    batch_results = await self._generate_image_batch(batch, kwargs) #这里会有长度77报错
                results.extend(batch_results)
            except RuntimeError as e:
                print(f"black box's generation failed: {str(e)}")
                results.extend([None]*len(batch))
                
        # print(f": {time.time()-start_time:.2f}s")
        return results
    
    async def _generate_image_batch(self, batch, kwargs):
        """"""

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        generator = torch.Generator(self.device).manual_seed(1234)
        # 
        if hasattr(self.pipe, '_encode_prompt'):
            with autocast(self.device):
                outputs = self.pipe(
                    prompt=batch,
                    num_images_per_prompt=1,
                    guidance_scale=15,
                    eta=0.0,
                    num_inference_steps=70,
                    generator=generator,
                    **kwargs
                )
                return outputs.images
        else:
            # 
            return await asyncio.gather(*[
                self._run_image_generation([prompt], kwargs) 
                for prompt in batch
            ])

    def _run_text_generation(self, batch: list, gen_kwargs: dict) -> list:
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]

    def _run_image_generation(self, batch: list, gen_kwargs: dict) -> list:
        results = []
        for prompt in batch:
            with autocast(self.pipe.device.type):
                image = self.pipe(prompt, **gen_kwargs).images[0]
            results.append(image)
        return results

# --------------------  --------------------
class MomentumOptimizer:
    """
     LoRA 
    """
    def __init__(self, whitebox: WhiteBoxModel, blackbox: BlackBoxModel, evaluator: Evaluator, args):
        self.whitebox = whitebox
        self.blackbox = blackbox
        self.evaluator = evaluator
        self.h = args.h
        self.n_directions = args.n_directions
        self.beta = args.beta
        self.velocity = torch.zeros_like(self.whitebox.get_flat_params())
        self.grad_history = []      # 
        self.grad_metadata = []     # 
        self.n_components = 50      # 
        self.args = args

    def _store_gradient(self, grad: torch.Tensor):
        """"""
        # 
        self.grad_history.append(grad.cpu().clone())
        
        # 
        stats = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'norm': torch.norm(grad).item(),
            'min': grad.min().item(),
            'max': grad.max().item()
        }
        self.grad_metadata.append(stats)
        
        # 
        if len(self.grad_history) > 100:
            self.grad_history.pop(0)
            self.grad_metadata.pop(0)
    def analyze_gradients(self, output_dir: str = "./grad_analysis"):
        """"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 
        self._plot_basic_stats(os.path.join(output_dir, "basic_stats.png"))
        
        # 
        self._analyze_dimensions(os.path.join(output_dir, "dimension_analysis.txt"))
        
        # 
        self._correlation_analysis(os.path.join(output_dir, "correlation_analysis.png"))
        
        # 
        self._direction_consistency(os.path.join(output_dir, "direction_heatmap.png"))

    def _plot_basic_stats(self, save_path: str):
        """"""
        
        # 
        steps = np.arange(len(self.grad_metadata))
        means = [m['mean'] for m in self.grad_metadata]
        stds = [m['std'] for m in self.grad_metadata]
        norms = [m['norm'] for m in self.grad_metadata]

        plt.figure(figsize=(15, 5))
        
        # 
        plt.subplot(1, 3, 1)
        plt.plot(steps, means, label='Mean')
        plt.title("Gradient Mean")
        plt.xlabel("Step")
        
        # 
        plt.subplot(1, 3, 2)
        plt.plot(steps, stds, color='orange', label='Std')
        plt.title("Gradient Std")
        plt.xlabel("Step")
        
        # 
        plt.subplot(1, 3, 3)
        plt.plot(steps, norms, color='green', label='Norm')
        plt.title("Gradient Norm")
        plt.xlabel("Step")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _analyze_dimensions(self, save_path: str):
        """"""
        # 
        grad_matrix = torch.stack(self.grad_history).numpy()  # [n_steps, n_dims]
        
        # 
        dim_means = np.mean(grad_matrix, axis=0)
        dim_stds = np.std(grad_matrix, axis=0)
        
        # 
        top10_indices = np.argsort(np.abs(dim_means))[-10:][::-1]
        
        # 
        with open(save_path, 'w') as f:
            f.write("=== Dimension-level analysis reports ===\n")
            f.write(f"Total Parameter Dimension: {grad_matrix.shape[1]}\n")
            f.write(f"Mean Gradient Absolute Value Mean: {np.mean(np.abs(dim_means)):.4f}\n")
            f.write(f"Maximum Gradient Mean Dimension: {top10_indices[0]} (值={dim_means[top10_indices[0]]:.4f})\n")
            f.write(f"Least stable dimension: {np.argmax(dim_stds)} (标准差={np.max(dim_stds):.4f})\n")
            
            f.write("\nTop 10 active dimension:\n")
            for idx in top10_indices:
                f.write(f"Dim {idx}: mean={dim_means[idx]:.4f}, std={dim_stds[idx]:.4f}\n")

    def _correlation_analysis(self, save_path: str):
        """"""
        import seaborn as sns
        
        # 
        sample_indices = np.random.choice(len(self.grad_history), size=50, replace=False)
        grad_samples = [self.grad_history[i].numpy() for i in sample_indices]
        
        # 
        cos_sim = np.zeros((len(grad_samples), len(grad_samples)))
        for i in range(len(grad_samples)):
            for j in range(len(grad_samples)):
                cos_sim[i,j] = np.dot(grad_samples[i], grad_samples[j]) / (
                    np.linalg.norm(grad_samples[i]) * np.linalg.norm(grad_samples[j]))
        
        # 
        plt.figure(figsize=(10, 8))
        sns.heatmap(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Gradient Direction Similarity")
        plt.savefig(save_path)
        plt.close()

    def _direction_consistency(self, save_path: str):
        """"""
        from sklearn.decomposition import PCA
        
        # 
        grad_matrix = torch.stack(self.grad_history).numpy()
        
        # 
        pca = PCA(n_components=self.n_components)
        pca.fit(grad_matrix)
        
        # 
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Analysis of Gradients')
        plt.savefig(save_path)
        plt.close()

    async def estimate_gradient(self, original_prompts: list) -> torch.Tensor:
        original_params = self.whitebox.get_flat_params()
        total_grad = torch.zeros_like(original_params)
        param_dim = original_params.shape[0]

        for _ in range(self.n_directions):
            # while True:
            delta = torch.randn(param_dim).to(original_params.device) * self.h
            # print(f" ： {delta}")
            # 
            # for name, param in self.whitebox.model.named_parameters():
            #     if 'lora_A' in name and 'ayers.0.self_attn' in name:
            #         print(f"Parameter {name} no set_flat_params: {param.data}")
                    
            self.whitebox.set_flat_params(original_params + delta)
            pos_prompts = self.whitebox.generate(original_prompts)
            # 
            self.whitebox.set_flat_params(original_params - delta)
            neg_prompts = self.whitebox.generate(original_prompts)
                # if pos_prompts != neg_prompts:
                #     break
                # print('')

            pos_results = await self.blackbox.generate(pos_prompts)
            if self.evaluator.eval_type == "image":
                pos_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(pos_prompts, pos_results)]
            else:
                pos_scores = [self.evaluator.evaluate(r) for r in pos_results]
            pos_score_avg = np.mean(pos_scores)
            print(f"pos_score_avg :{pos_score_avg}")

            neg_results = await self.blackbox.generate(neg_prompts)
            if self.evaluator.eval_type == "image":
                neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
            else:
                neg_scores = [self.evaluator.evaluate(r) for r in neg_results]
            neg_score_avg = np.mean(neg_scores)
            print(f"neg_score_avg :{neg_score_avg}")
            score_diff = pos_score_avg - neg_score_avg
            print(f"n_grad :{(score_diff / (2 * self.h)) * delta}")
            total_grad += (score_diff / (2 * self.h)) * delta

        # 
        self.whitebox.set_flat_params(original_params)
        final_grad = total_grad / self.n_directions
        print(f"final_grad :{final_grad}")
        # self._store_gradient(final_grad)  # 
        return final_grad
    async def spsa_gradient(self, original_prompts: list) -> torch.Tensor:
        # 
        original_params = self.whitebox.get_flat_params()
        total_grad = torch.zeros_like(original_params)
        param_dim = original_params.shape[0]

        for _ in range(self.n_directions):
            

            delta = torch.randint(0, 2, (param_dim,), device=original_params.device, dtype=torch.float32) * 2 - 1
            # 
            self.whitebox.set_flat_params(original_params + self.h * delta)
            pos_prompts = self.whitebox.generate(original_prompts)

            # 
            self.whitebox.set_flat_params(original_params - self.h * delta)
            neg_prompts = self.whitebox.generate(original_prompts)

            pos_results = await self.blackbox.generate(pos_prompts)
            if self.evaluator.eval_type == "image":
                pos_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(pos_prompts, pos_results)]
            else:
                pos_scores = [self.evaluator.evaluate(r) for r in pos_results]
            pos_score_avg = np.mean(pos_scores)
            print(f"pos_score_avg :{pos_score_avg}")

            neg_results = await self.blackbox.generate(neg_prompts)
            if self.evaluator.eval_type == "image":
                neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
            else:
                neg_scores = [self.evaluator.evaluate(r) for r in neg_results]
            neg_score_avg = np.mean(neg_scores)
            print(f"neg_score_avg :{neg_score_avg}")
            # 
            score_diff = pos_score_avg - neg_score_avg

            # SPSA 
            #  : (score_diff)/(2*h*delta[i]) delta[i] ∈ {+1, -1}，
            #  (score_diff)/(2*h) * delta
            print(f"n_grad :{(score_diff / (2 * self.h)) * delta}")
            total_grad += (score_diff / (2 * self.h)) * delta

        # 
        self.whitebox.set_flat_params(original_params)
        final_grad = total_grad / self.n_directions
        print(f"final_grad :{final_grad}")
        # self._store_gradient(final_grad)  # 
        return final_grad


    def update_params(self, gradient: torch.Tensor, lr: float = 0.01):
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient
        new_params = self.whitebox.get_flat_params() + lr * self.velocity
        self.whitebox.set_flat_params(new_params)



#--------------------- ------------------------
async def main(args):
    logger = TrainingLogger(args)
    test_dataset = Dataset(f"./dataset/{args.dataset}_train.csv") #256
    train_dataset = Dataset(f"./dataset/{args.dataset}_test.csv") #544
    
    # _, train_dataset = train_test_split(train_dataset, test_size=0.4706, random_state=args.seed) #256
    # _, train_dataset = train_test_split(train_dataset, test_size=0.2353, random_state=args.seed) #128
    test_size=args.train_samples/544
    _, train_dataset = train_test_split(train_dataset, test_size=test_size, random_state=args.seed) #64
    if args.debug == 'True':
        _, test_dataset = train_test_split(test_dataset, test_size=0.0625, random_state=args.seed) #16 
    print(f"{len(train_dataset)}: train samples, {len(test_dataset)}: test samples")
    batch_size = getattr(args, "batch_size", 1)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 
    demos_template = "Original: [INPUT]\nRephrase: [OUTPUT]"
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data[args.example])
    init_prompt = ['\n']
    if args.ptype == 0:
        prompt_gen_str = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, rephrase this sentence with consistency guaranteed:\n[original_prompt]\n"
    elif args.ptype == 1:
        prompt_gen_str = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 2:
        prompt_gen_str = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, using your creativity to rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 3:
        prompt_gen_str = "[full_DEMO]\n\nIn order to make the diffusion model generate better pictures, based on the rephrasing way in the examples above, rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 4:
        prompt_gen_str = "[full_DEMO]\n\nIn order to make the diffusion model generate more beautiful pictures, based on the rephrasing way in the examples above, using your creativity rather than just applying the example content to rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 5:
        prompt_gen_str = "[full_DEMO]\n\nIn order to make the diffusion model generate better pictures, based on the rephrasing way in the examples above, using your creativity rather than just applying the example content to rephrase this sentence:\n[original_prompt]\n"
        # prompt_gen_str = "In order to make the diffusion model generate better pictures, based on the rephrasing way in the examples above, using your creativity rather than just applying the example content to rephrase this sentence:\n[original_prompt]\n"
    prompt_gen_template = template.InitQATemplate(prompt_gen_str)
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The user gives a few examples of rephrasing and a sentence that needs to be rephrased. The assistant provides a rephrased sentence without additional content for user."
    # device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    device = f"cuda:{args.cuda}"
    if os.path.exists(args.output_dir + '/' + "adapter_config.json"):
        lora_resume = True
    else:
        lora_resume = False
    # whitebox = WhiteBoxModel("meta-llama/Llama-2-7b-hf", hf_token="your huggingface token")
    whitebox = WhiteBoxModel(model_name=args.white_model, device=device, hf_token=args.hf_token if hasattr(args, 'hf_token') else None, lora_rank=args.lora_rank) 
    blackbox_mode = getattr(args, "blackbox_mode", "image")
    if blackbox_mode == "text":
        blackbox = BlackBoxModel(model_type="text", model_name=args.black_model, device=device, batch_size=batch_size)
        reference_text = "LLM stands for large language model."
        evaluator = Evaluator("text", reference=reference_text, args = args)
    elif blackbox_mode == "image":
        blackbox = BlackBoxModel(model_type="image", model_name=args.black_model, device=device, batch_size=batch_size)
        evaluator = Evaluator("image", cache_dir="./cache/", device=device, args = args)
    else:
        raise ValueError("Invalid blackbox_mode")

    optimizer = MomentumOptimizer(whitebox, blackbox, evaluator, args)


    # initial_model_state = whitebox.model.state_dict()


    async def evaluate_model(dataloader, evaluator,epoch):
        scores = []
        sub_scores = {'aesthetic': [], 'clip': [], 'pick': []}
        
        #for batch in dataloader:
        for batch_idx, batch in enumerate(dataloader):
            original_prompts = batch['text']
            
            # 
            if args.white_model in ["promtist", "sft"]:
                input_text = original_prompts
            else:
                text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
            generated_prompts = whitebox.generate(input_text)
                
            results = await blackbox.generate(generated_prompts)
            if args.gene_image == 'True' and epoch != -2:
                save_batch_results(
                    results,
                    generated_prompts,
                    args.output_dir,
                    epoch,
                    method="ours",
                    batch_idx=batch_idx
                )
            if args.gene_image == 'True' and epoch == -2:
                save_batch_results(
                    results,
                    generated_prompts,
                    args.output_dir,
                    epoch,
                    method="ablation",
                    batch_idx=batch_idx
                )
            
            batch_scores = []
            for prompt, result in zip(generated_prompts, results):
                if evaluator.eval_type == "image":
                    score_dict = evaluator.evaluate(prompt, result)
                    batch_scores.append(score_dict[args.metric])
                    for k in sub_scores:
                        sub_scores[k].append(score_dict[k])
                else:
                    score = evaluator.evaluate(result)
                    batch_scores.append(score)
            
            scores.extend(batch_scores)
            
        return np.array(scores), sub_scores if evaluator.eval_type=="image" else None
    
    async def original_test(dataloader, evaluator):
        scores = []
        sub_scores = {'aesthetic': [], 'clip': [], 'pick': []}
        
        #for batch in dataloader:
        for batch_idx, batch in enumerate(dataloader):
            original_prompts = batch['text']
            results = await blackbox.generate(original_prompts)
            if args.gene_image == 'True':
                save_batch_results(
                    results,
                    original_prompts,
                    args.output_dir,
                    epoch=-1,
                    method="ori",
                    batch_idx=batch_idx
                )
            
            batch_scores = []
            for prompt, result in zip(original_prompts, results):
                if evaluator.eval_type == "image":
                    score_dict = evaluator.evaluate(prompt, result)
                    batch_scores.append(score_dict[args.metric])
                    for k in sub_scores:
                        sub_scores[k].append(score_dict[k])
                else:
                    score = evaluator.evaluate(result)
                    batch_scores.append(score)
            
            scores.extend(batch_scores)
            
        return np.array(scores), sub_scores if evaluator.eval_type=="image" else None
    
    print("--------------------------original prompt test---------------------------")
    #     optimizer.analyze_gradients(args.output_dir / f"{epoch}")
        

            
    test_scores, subscore = await original_test(test_dataloader, evaluator)
    mean_subscore = {key: np.mean(values) for key, values in subscore.items()}
    print(f"subscore{mean_subscore}")
    logger.log_training_step({
        'epoch': "original",
        'test_avg_score': np.mean(test_scores),
        'mean_subscore': mean_subscore,
    })
    print("--------------------------original prompt finish---------------------------")
    
    print("---------------------------Perform initial testing------------------------------")
    orig_test_scores, orig_subscore = await evaluate_model(test_dataloader, evaluator,epoch=-2)
    mean_subscore = {key: np.mean(values) for key, values in orig_subscore.items()}
    # for name, param in whitebox.model.named_parameters():
    #     if 'lora_A' in name or 'lora_B' in name:
    #         print(f"Parameter {name} set_flat_params: {param.data}")
    print(f"subscore{mean_subscore}")
    logger.log_training_step({
        'epoch': "ablation",
        'orig_score': np.mean(orig_test_scores),
        'orig_subscore': mean_subscore,
    })
    print("---------------------------initial testing finished------------------------------")
    
    if lora_resume != True: 
        if args.batch_mode == "batch":

            print("------------------------Batch mode training-----------------------") 
            for epoch in range(args.epochs):
                epoch_start = time.time()
                
                for batch_idx, batch in enumerate(train_dataloader):
                    original_prompts = batch['text']
                    print(f"batch_idx: {batch_idx}")
                    # 
                    if args.white_model in ["promtist", "sft"]:
                        input_text = original_prompts
                    else:
                        text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                        input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                    
                    if args.optimizer == "mmt":
                        grad = await optimizer.estimate_gradient(input_text)
                    if args.optimizer == "spsa":
                        grad = await optimizer.spsa_gradient(input_text)
                    optimizer.update_params(grad, lr=args.lr)

                # optimizer.analyze_gradients(args.output_dir / f"{epoch}")
                
                for name, param in whitebox.model.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name:
                        print(f"epoch: {epoch} , Parameter {name} set_flat_params: {param.data}")
                # 
                test_scores, subscore = await evaluate_model(test_dataloader, evaluator,epoch)
                mean_subscore = {key: np.mean(values) for key, values in subscore.items()}
                print(f"subscore{mean_subscore}")
                logger.log_training_step({
                    'epoch': epoch,
                    'test_avg_score': np.mean(test_scores),
                    'mean_subscore': mean_subscore,
                    'epoch_time': time.time() - epoch_start
                })
                epoch_save_LoRA_dir=os.path.join(args.output_dir,f"_epoch{epoch}")
                os.makedirs(epoch_save_LoRA_dir,exist_ok=True)
                whitebox.model.save_pretrained(epoch_save_LoRA_dir)
                print(f"save Epoch {epoch} 's LoRA's state to {epoch_save_LoRA_dir}")
        else:
            print("------------------------Sampling pattern training-----------------------") 
            for batch_idx, batch in enumerate(train_dataloader):
                original_prompts = batch['text']
                # 
                if args.white_model in ["promtist", "sft"]:
                    input_text = original_prompts
                else:
                    text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                    input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]

                for epoch in range(args.epochs):
                    print(f"Epoch {epoch+1}/{args.epochs}")
                    start_time = time.time()
                
                    # 
                    if args.optimizer == "mmt":
                        grad = await optimizer.estimate_gradient(input_text)
                    if args.optimizer == "spsa":
                        grad = await optimizer.spsa_gradient(input_text)
                    optimizer.update_params(grad, lr=args.lr)

                    #
                    if epoch % 5 == 0:
                        generated_prompts = whitebox.generate(input_text)
                        results = await blackbox.generate(generated_prompts)
                        score_dict = evaluator.evaluate(generated_prompts[0], results[0])
                        logger.log_training_step({
                            'text': original_prompts,
                            'local_epoch': epoch,
                            'current_score': score_dict,
                            'time_spent': time.time() - start_time
                        })
                        
        whitebox.model.save_pretrained(args.output_dir)
        print(f"Saved LoRA parameters to {args.output_dir}")
        # logger.finalize()
        # for name, param in whitebox.model.named_parameters():
        #     if 'lora_A' in name or 'lora_B' in name:
        #         print(f"Parameter {name} set_flat_params: {param.data}")
        print("finish training")
        if args.soft_train == 'True':
             #--------------
            print("-------------------------Start soft prompt optimization-------------------------")
            
            # Make sure to reload the model FIRST if it's a resume scenario
            if lora_resume:
                print(f"Loading existing LoRA from {args.output_dir}")
                whitebox.load_lora(args.output_dir)
                whitebox.merge_lora()

                # 
                whitebox.model.eval()
            
            # 
            soft_prompt_optimizer = SoftPromptOptimizer(whitebox, blackbox, evaluator, args)
            
            # 
            async def evaluate_with_soft_prompt(dataloader, evaluator, soft_prompt_embeds=None):
                scores = []
                sub_scores = {'aesthetic': [], 'clip': [], 'pick': []}
                
                for batch_idx, batch in enumerate(dataloader):
                    original_prompts = batch['text']
                    
                    # 
                    if args.white_model in ["promtist", "sft"]:
                        input_text = original_prompts
                    else:
                        text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                        input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                    generated_prompts = whitebox.generate(input_text, soft_prompt_embeds=soft_prompt_embeds)

                    results = await blackbox.generate(generated_prompts)
                    
                    # 
                    if args.gene_image == 'True' and epoch is not None:
                        save_batch_results(
                            results,
                            generated_prompts,
                            args.soft_output_dir,
                            epoch=epoch,
                            method="soft_prompt_eval",
                            batch_idx=batch_idx
                        )
                    
                    batch_scores = []
                    for prompt, result in zip(generated_prompts, results):
                        if evaluator.eval_type == "image":
                            score_dict = evaluator.evaluate(prompt, result)
                            batch_scores.append(score_dict[args.metric])
                            for k in sub_scores:
                                sub_scores[k].append(score_dict[k])
                        else:
                            score = evaluator.evaluate(result)
                            batch_scores.append(score)
                    
                    scores.extend(batch_scores)
                    
                return np.array(scores), sub_scores if evaluator.eval_type=="image" else None
            '''
            # 
            print("...")
            base_test_scores, base_subscore = await evaluate_with_soft_prompt(test_dataloader, evaluator)
            
            if isinstance(base_subscore, dict):
                base_mean_subscore = {key: np.mean(values) for key, values in base_subscore.items()}
                print(f": {base_mean_subscore}")
            
            base_mean_score = np.mean(base_test_scores)
            print(f": {base_mean_score:.4f}")
            
            logger.log_training_step({
                'epoch': "soft_prompt_base",
                'base_score': base_mean_score,
                'base_subscore': base_mean_subscore if isinstance(base_subscore, dict) else None,
            })
            '''
            # 
            if args.batch_mode == "batch":
                print("------------------------Batch mode soft prompt training-----------------------")
                for epoch in range(args.soft_epochs):
                    epoch_start = time.time()
                    epoch_scores = []
                    
                    for batch_idx, batch in enumerate(train_dataloader):
                        original_prompts = batch['text']
                        print(f"Soft batch_idx: {batch_idx}")
                        
                        # 
                        if args.white_model in ["promtist", "sft"]:
                            input_text = original_prompts
                        else:
                            text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                            input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                        
                        # 
                        result = await soft_prompt_optimizer.optimize_step(
                            input_text, 
                            batch_idx=batch_idx, 
                            epoch=epoch,
                            output_dir=args.soft_output_dir
                        )
                        
                        # 
                        # epoch_scores.append(result["score"])
                        # logger.log_training_step({
                        #     'epoch': f"soft_epoch_{epoch}",
                        #     'batch': batch_idx,
                        #     'score': result["score"],
                        #     'batch_time': time.time() - epoch_start
                        # })
                    
                    #  epoch 
                    print(f"Soft Prompt Epoch {epoch} evaluate...")
                    current_embeds = soft_prompt_optimizer.get_soft_prompt_embeds()
                    eval_scores, eval_subscore = await evaluate_with_soft_prompt(
                        test_dataloader, 
                        evaluator, 
                        soft_prompt_embeds=current_embeds
                    )
                    
                    if isinstance(eval_subscore, dict):
                        eval_mean_subscore = {key: np.mean(values) for key, values in eval_subscore.items()}
                        print(f"Epoch {epoch} score : {eval_mean_subscore}")
                    
                    eval_mean_score = np.mean(eval_scores)
                    print(f"Epoch {epoch} average score : {eval_mean_score:.4f}")
                    
                    #  epoch 
                    logger.log_training_step({
                        'epoch': f"soft_eval_{epoch}",
                        'eval_score': eval_mean_score,
                        'eval_subscore': eval_mean_subscore if isinstance(eval_subscore, dict) else None,
                        'epoch_time': time.time() - epoch_start,
                        'epoch_avg_batch_score': np.mean(epoch_scores)
                    })
                    
                # 
                soft_prompt_dir = os.path.join(args.soft_output_dir, "soft_prompt")
                soft_prompt_optimizer.save(soft_prompt_dir)
                
            else:
                # 
                print("------------------------Whole sample soft cue training-----------------------")
                
                # 
                all_prompts = []
                batches_collected = 0
                
                # 
                for batch in train_dataloader:
                    batch_prompts = batch['text']
                    all_prompts.extend(batch_prompts)
                    batches_collected += 1
                    if batches_collected >= args.soft_train_batches:
                        break
                
                if args.white_model in ["promtist", "sft"]:
                    input_text = all_prompts
                else:
                    text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in all_prompts]
                    input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                
                print(f"use {len(all_prompts)} Soft prompts optimization for each sample...")
                
                # 
                optimization_result = await soft_prompt_optimizer.optimize(
                    input_text, 
                    args.soft_epochs,
                    output_dir=args.soft_output_dir
                )
                
                # 
                soft_prompt_dir = os.path.join(args.soft_output_dir, "soft_prompt")
                soft_prompt_optimizer.save(soft_prompt_dir)
    elif lora_resume == True and args.soft_train == 'False': #lora reusme
        print(f"Loading existing LoRA from {args.output_dir}")
        whitebox.load_lora(args.output_dir)
        whitebox.merge_lora()

        # 
        whitebox.model.eval()

        async def evaluate_model(dataloader, evaluator):
            scores = []
            sub_scores = {'aesthetic': [], 'clip': [], 'pick': []}
            
            for batch in dataloader:
                original_prompts = batch['text']
                
                # 
                if args.white_model in ["promtist", "sft"]:
                    input_text = original_prompts
                else:
                    text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                    input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                generated_prompts = whitebox.generate(input_text)
                # print("-------------------evaluate_model--------------------")
                # for name, param in whitebox.model.named_parameters():
                #     if 'lora_A' in name or 'lora_B' in name:
                #         print(f"Parameter {name} set_flat_params: {param.data}")

                results = await blackbox.generate(generated_prompts)
                
                batch_scores = []
                for prompt, result in zip(generated_prompts, results):
                    if evaluator.eval_type == "image":
                        score_dict = evaluator.evaluate(prompt, result)
                        batch_scores.append(score_dict[args.metric])
                        for k in sub_scores:
                            sub_scores[k].append(score_dict[k])
                    else:
                        score = evaluator.evaluate(result)
                        batch_scores.append(score)
                
                scores.extend(batch_scores)
                
            return np.array(scores), sub_scores if evaluator.eval_type=="image" else None
        
        print("---------------------------Perform heavy load tests------------------------------")
        resume_scores, resume_subscore = await evaluate_model(test_dataloader, evaluator)
        resume_mean_subscore = {key: np.mean(values) for key, values in resume_subscore.items()}
        # for name, param in whitebox.model.named_parameters():
        #     if 'lora_A' in name or 'lora_B' in name:
        #         print(f"Parameter {name} set_flat_params: {param.data}")
        print(f"subscore{mean_subscore}")
        logger.log_training_step({
            'epoch': "resume",
            'resume_score': np.mean(resume_scores),
            'resume_subscore': resume_mean_subscore,
        })
        print("---------------------------heavy load tests finished------------------------------")
    else :#lora_resume == True and args.soft_train == 'True':
         #--------------
        print("-------------------------strat soft prompt optimization-------------------------")
        
        # Make sure to reload the model FIRST if it's a resume scenario
        
        if lora_resume:
            print(f"Loading existing LoRA from {args.output_dir}")
            whitebox.load_lora(args.output_dir)
            whitebox.merge_lora()

            # 
            whitebox.model.eval()
        
        # 
        soft_prompt_optimizer = SoftPromptOptimizer(whitebox, blackbox, evaluator, args)
        
        # 
        async def evaluate_with_soft_prompt(dataloader, evaluator, soft_prompt_embeds=None,epoch=None):
            scores = []
            sub_scores = {'aesthetic': [], 'clip': [], 'pick': []}
            
            for batch_idx, batch in enumerate(dataloader):
                original_prompts = batch['text']
                
                # 
                if args.white_model in ["promtist", "sft"]:
                    input_text = original_prompts
                else:
                    text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                    input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                generated_prompts = whitebox.generate(input_text, soft_prompt_embeds=soft_prompt_embeds)

                results = await blackbox.generate(generated_prompts)
                
                # 
                if args.gene_image == 'True' and epoch is not None:
                    save_batch_results(
                        results,
                        generated_prompts,
                        args.soft_output_dir,
                        epoch=epoch,
                        method="soft_prompt_eval",
                        batch_idx=batch_idx
                    )
                
                batch_scores = []
                for prompt, result in zip(generated_prompts, results):
                    if evaluator.eval_type == "image":
                        score_dict = evaluator.evaluate(prompt, result)
                        batch_scores.append(score_dict[args.metric])
                        for k in sub_scores:
                            sub_scores[k].append(score_dict[k])
                    else:
                        score = evaluator.evaluate(result)
                        batch_scores.append(score)
                
                scores.extend(batch_scores)
                
            return np.array(scores), sub_scores if evaluator.eval_type=="image" else None
        
        # 
        print("Perform benchmarking (without soft prompts)...")
        base_test_scores, base_subscore = await evaluate_with_soft_prompt(test_dataloader, evaluator)
        
        if isinstance(base_subscore, dict):
            base_mean_subscore = {key: np.mean(values) for key, values in base_subscore.items()}
            print(f"base subscore: {base_mean_subscore}")
        
        base_mean_score = np.mean(base_test_scores)
        print(f"Benchmark average score: {base_mean_score:.4f}")
        
        logger.log_training_step({
            'epoch': "soft_prompt_base",
            'base_score': base_mean_score,
            'base_subscore': base_mean_subscore if isinstance(base_subscore, dict) else None,
        })
        
        # Check for soft-prompt optimization in batch mode
        if args.batch_mode == "batch":
            print("------------------------Batch mode soft cue training-----------------------")
            for epoch in range(args.soft_epochs):
                epoch_start = time.time()
                epoch_scores = []
                
                for batch_idx, batch in enumerate(train_dataloader):
                    original_prompts = batch['text']
                    print(f"Soft batch_idx: {batch_idx}")
                    
                    # Using white box models to generate optimized prompts
                    if args.white_model in ["promtist", "sft"]:
                        input_text = original_prompts
                    else:
                        text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                        input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                    
                    # Perform one soft-prompt optimization step
                    result = await soft_prompt_optimizer.optimize_step(
                        input_text, 
                        batch_idx=batch_idx, 
                        epoch=epoch,
                        output_dir=args.soft_output_dir
                    )
                    
                    # Record the results of each batch
                    epoch_scores.append(result["score"])
                    # logger.log_training_step({
                    #     'epoch': f"soft_epoch_{epoch}",
                    #     'batch': batch_idx,
                    #     'score': result["score"],
                    #     'batch_time': time.time() - epoch_start
                    # })
                
                # Evaluated at the end of each epoch
                print(f"Soft Prompt Epoch {epoch} 评估...")
                current_embeds = soft_prompt_optimizer.get_soft_prompt_embeds()
                eval_scores, eval_subscore = await evaluate_with_soft_prompt(
                    test_dataloader, 
                    evaluator, 
                    soft_prompt_embeds=current_embeds,
                    epoch=epoch
                )
                
                if isinstance(eval_subscore, dict):
                    eval_mean_subscore = {key: np.mean(values) for key, values in eval_subscore.items()}
                    print(f"Epoch {epoch} subfraction: {eval_mean_subscore}")
                
                eval_mean_score = np.mean(eval_scores)
                print(f"Epoch {epoch} average score: {eval_mean_score:.4f}")
                
                # Record epoch assessment results
                logger.log_training_step({
                    'epoch': f"soft_eval_{epoch}",
                    'eval_score': eval_mean_score,
                    'eval_subscore': eval_mean_subscore if isinstance(eval_subscore, dict) else None,
                    'epoch_time': time.time() - epoch_start,
                    'epoch_avg_batch_score': np.mean(epoch_scores)
                })
                
            # Save the final soft tip
            soft_prompt_dir = os.path.join(args.soft_output_dir, "soft_prompt")
            soft_prompt_optimizer.save(soft_prompt_dir)
            
        else:
            # Default mode: one-time optimization using all samples
            print("------------------------Whole sample soft cue training-----------------------")
            
            # 
            all_prompts = []
            batches_collected = 0
            
            # 
            for batch in train_dataloader:
                batch_prompts = batch['text']
                all_prompts.extend(batch_prompts)
                batches_collected += 1
                if batches_collected >= args.soft_train_batches:
                    break
            
            if args.white_model in ["promtist", "sft"]:
                input_text = all_prompts
            else:
                text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in all_prompts]
                input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
            
            print(f"Soft cue optimization using {len(all_prompts)} samples...")
            
            # 
            optimization_result = await soft_prompt_optimizer.optimize(
                input_text, 
                args.soft_epochs,
                output_dir=args.soft_output_dir
            )
            
            # 
            soft_prompt_dir = os.path.join(args.soft_output_dir, "soft_prompt")
            soft_prompt_optimizer.save(soft_prompt_dir)



if __name__ == "__main__":
    args = parse_args()
    set_all_seed(args.seed)
    asyncio.run(main(args))

