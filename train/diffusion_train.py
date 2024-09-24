import sys
sys.path.append("/TG-CDDPM-main")

import argparse
import torch.cuda
from transformers import AutoTokenizer
from train.TexPepAlignment import TextEncoder, PepEncoder
from config.backen_config import ProGenConfig
from model.backend import FacModel
from model.resample import create_named_schedule_sampler
from utils.tokenizer import load_data
from utils.script_utils import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from utils.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("creating model and diffusion...")

    # diffusion model building
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    model.load_state_dict(torch.load(args.pretrained_diffusion))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # peptide and text encoder building
    pep_config = ProGenConfig(vocab_size=30,
                              n_positions=args.n_positions,
                              n_ctx=args.n_ctx,
                              n_embd=args.n_embd,
                              n_layer=2)
    text_encoder = TextEncoder()
    text_encoder.load_state_dict(torch.load(args.pretrained_text_encoder))
    text_encoder.to(device)
    pep_encoder = PepEncoder(pep_config)
    pep_encoder.load_state_dict(torch.load(args.pretrained_pep_encoder))
    pep_encoder.to(device)

    # translator
    translator = FacModel(pep_config)
    translator.load_state_dict(torch.load(args.pretrained_translator))
    translator.to(device)
    # facilitator, _ = create_model_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys())
    # )
    # facilitator.load_state_dict(torch.load(args.checkpoint_dir + "/diffusion_model_facilitator.pt"))
    # facilitator.to(device)

    print("creating data loader...")
    # myTokenizer = tokenizer(args.vocab_path)
    myTokenizer = AutoTokenizer.from_pretrained('../checkpoints/bert/prot_bert')
    data = load_data(myTokenizer, args.seq_len, "train", args)

    print("training...")
    TrainLoop(
        model=model,
        translator=translator,
        diffusion=diffusion,
        text_encoder=text_encoder,
        pep_encoder=pep_encoder,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        # path of dataset
        data_dir="../dataset",
        val_data_dir="../dataset",
        vocab_path='../mapping/vocab.txt',
        # path of pretrained model
        checkpoint_dir='../checkpoints',
        pretrained_diffusion='../checkpoints/diffusion_model_pre_trained.pt',
        pretrained_text_encoder='../checkpoints/text_encoder_4.pt',
        pretrained_pep_encoder='../checkpoints/pep_encoder_4.pt',
        pretrained_translator='../checkpoints/diffusion_model_facilitator.pt',
        timesteps=500,
        train_epoches=1000,
        lr=5e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        seq_len=50,
        is_need_classifier=False,
        label_num=2,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler="uniform",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
