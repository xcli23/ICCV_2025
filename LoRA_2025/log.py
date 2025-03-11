import os
import json
import time
from datetime import datetime
class TrainingLogger:
    """"""
    def __init__(self, args):
        # 
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path=args.soft_output_dir#
        self.seed=args.seed
        
        # 
        self.metadata = {
            'fixed_params': {
                'dataset': args.dataset,
                'train_samples': args.train_samples,
                'white_model': args.white_model,
                'black_model': args.black_model,
                'dataset': args.dataset,
                'optimizer': args.optimizer,
                'h': args.h,
                'lr': args.lr,
                'lora_rank': args.lora_rank,
                'metric': args.metric,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'n_directions': args.n_directions,
                'lambda_aes': args.lambda_aes,
                'lambda_clip': args.lambda_clip,
                'lambda_pick': args.lambda_pick,
                'output_dir': args.output_dir,
                'seed': args.seed,
                # 
                'soft_output_dir':args.soft_output_dir,
                'soft_train': args.soft_train,
                'soft_epochs': args.soft_epochs,
                'soft_lr': args.soft_lr,
                'mu': args.mu,
                'intrinsic_dim': args.intrinsic_dim,
                'n_prompt_tokens': args.n_prompt_tokens,
                'soft_train_batches': args.soft_train_batches,
                'random_proj': args.random_proj,
                'soft_n_directions':args.soft_n_directions,
                'debug': args.debug,
                'example': args.example,
                'ptype': args.ptype
            },
            'dynamic_logs': []
        }
        
        # 
        self._save_metadata()

    def _save_metadata(self):
        """"""
        with open(os.path.join(self.output_path, f"seed{self.seed}_train.jsonl"), 'w') as f:
            json.dump(self.metadata['fixed_params'], f, indent=2)

    def log_training_step(self, log_data):
        """"""
        # 
        log_data['timestamp'] = datetime.now().isoformat()
        self.metadata['dynamic_logs'].append(log_data)
        
        # 
        with open(os.path.join(self.output_path, f"seed{self.seed}_train.jsonl"), 'a') as f:
            f.write(json.dumps(log_data) + "\n")

    def finalize(self):
        full_data = {
            'metadata': self.metadata['fixed_params'],
            'logs': self.metadata['dynamic_logs']
        }
        with open(os.path.join(self.output_path, "complete_report.json"), 'w') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)