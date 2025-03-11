import argparse
import random
import numpy as np
import torch
def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set all the seeds to {seed} successfully!")

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--batch_mode",type=str,default='batch')
    parser.add_argument("--white_model",type=str,default='vicuna-13b',help="The model name of the open-source LLM.")
    parser.add_argument("--black_model",type=str,default='sd1.5', help='he model name of the close-source LLM.')
    parser.add_argument("--blackbox_mode",type=str,default='image', help='text or image')
    # parser.add_argument("--input_file", type=str, default="./dataset/DiffusionDB_prompts_256_unique.tsv")
    parser.add_argument("--dataset", type=str, default="paintings",help="")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="spsa", help = "mmt or spsa")
    parser.add_argument("--metric", type=str, default="total", help = "aesthetic, clip, pick, total")
    parser.add_argument('--ptype', type=int,default=5)
    parser.add_argument('--example', type=str,default='promptist', help="promtist or beautiful")
    parser.add_argument("--lambda_aes", type=float, default=0.33)
    parser.add_argument("--lambda_clip", type=float, default=0.33) 
    parser.add_argument("--lambda_pick", type=float, default=0.34)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--soft_output_dir", type=str, default="./output")
    parser.add_argument("--gene_image", type=str, default="False")
    parser.add_argument("--debug", type=str, default="False")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate of optimizer")
    parser.add_argument("--h", type=float, default=0.05, help="The parameter of disturbance")
    parser.add_argument("--beta", type=float, default=0.5, help="The parameter of SPSA optimizer")
    parser.add_argument("--epochs", type=int, default=20, help="The number of epochs")
    parser.add_argument("--n_directions", type=int, default=20, help="The number of directions")
    parser.add_argument("--soft_n_directions", type=int, default=20, help="The number of directions")
    parser.add_argument("--train_samples", type=int, default=16, help="The number of data samples")
    parser.add_argument('--lora_rank', type=int, default=4,help='Rank of LoRA adaptation')
    parser.add_argument('--soft_train_batches', type=int, default=32, help='Number of batches to use for soft prompt optimization')
    parser.add_argument("--soft_train", type=str, default="False",help="")
    parser.add_argument("--soft_lr", type=float, default=0.1, help="")
    parser.add_argument("--soft_epochs", type=int, default=50, help="")
    parser.add_argument("--mu", type=float, default=0.1,help="ZO-OGD")
    parser.add_argument("--intrinsic_dim", type=int, default=10,help="")
    parser.add_argument("--n_prompt_tokens", type=int, default=5,help="")
    parser.add_argument("--random_proj", type=str, default="uniform",choices=["normal", "uniform"],help="")
    args = parser.parse_args()
    
    if args.batch_size == 1:
        args.batch_mode = "single"
    #
    if args.black_model == "sd1.5":
        args.black_model = "sd-legacy/stable-diffusion-v1-5"
    elif args.black_model == "sd1.4":
        args.black_model = "CompVis/stable-diffusion-v1-4"
    elif args.black_model == "sdXL":
        args.black_model = "stabilityai/stable-diffusion-xl-base-1.0"
    elif args.black_model == "dreamlike":
        args.black_model = "dreamlike-art/dreamlike-photoreal-2.0"
        #dreamlike-art/dreamlike-photoreal-2.0
    #
    if args.white_model == "llama2-7b":
        args.white_model = "meta-llama/Llama-2-7b-hf"
        args.hf_token = "your huggingface token"#Upload your huggingface token here
    elif args.white_model == "vicuna-13b":
        args.white_model = "lmsys/vicuna-13b-v1.3"
        args.hf_token = "your huggingface token"#Upload your huggingface token here
    elif args.white_model == "vicuna-7b":
        args.white_model = "lmsys/vicuna-7b-v1.5"
        args.hf_token = "your huggingface token"#Upload your huggingface token here

    args.device = torch.device("cuda", args.cuda)

    return args