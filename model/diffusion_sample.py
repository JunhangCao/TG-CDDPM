"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse

import jsonlines
import torch
import torch as th

from amPEP.amPEPpy.amPEP import amp_score
from utils.tokenizer import tokenizer, load_data
from utils.script_utils import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    device = ("cuda" if th.cuda.is_available() else "cpu")
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    myTokenizer = tokenizer(args.vocab_path)
    print("sampling...")
    for length in args.seq_len:
        all_images = []
        test_data = load_data(myTokenizer, length, 'train', args.cls, args.batch_size)
        while len(all_images) < args.num_samples:
            batch, cond = next(test_data)
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=1, size=(args.batch_size * 2,), device=device
                )
                classes = classes.reshape(args.batch_size, 2)
                # classes = torch.tensor()
                model_kwargs["label"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, length, args.n_embd),
                noise=torch.randn_like(model.get_input_embeddings()(cond['input_ids'].to(device)).to(device)),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = model.get_logits(sample)
            _, indices = torch.topk(sample, k=1, dim=-1)
            sampled_seq = myTokenizer.decode_batch_token(indices)
            all_images.extend(s for s in sampled_seq)
            print(f"created length {length} {args.batch_size} samples")
        with jsonlines.open('../sample/random.jsonl', 'w') as file:
            for seq in all_images:
                temp = {}
                # decoded_seq = myTokenizer.decode_token(seq).replace(' ','')
                temp['trg'] = seq
                temp['act'] = amp_score(seq[0])
                # file.write(temp)
                print(seq)
                print(amp_score(seq[0]))
                # print(evaluate(seq[0]))
        print(f"{length}sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=3,
        batch_size=1,
        max_loop=400,
        use_ddim=False,
        vocab_path="../mapping/vocab.txt",
        model_path="../checkpoints/diffusion_model010000.pt",
        classifier_scale=1.0,
        use_fp16=False,
        classifier_use_fp16=False,
        seq_len=[50],
        vocab_size=21,
        class_cond=False,
        cls=2

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
