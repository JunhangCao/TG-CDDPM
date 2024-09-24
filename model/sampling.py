import sys
sys.path.append("/TG-CDDPM-main")
import argparse
import os
import jsonlines
import torch
import torch as th
from transformers import AutoTokenizer
from config.backen_config import ProGenConfig
from train.TexPepAlignment import TextEncoder, PepEncoder
from utils.tokenizer import TextTokenizer
from model.backend import FacModel
from utils.script_utils import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

class PretainedTokenizer():
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('../checkpoints/bert/prot_bert')
        self.vocab = {}
        self.rev_vocab = {}
        with open('../checkpoints/bert/prot_bert/vocab.txt', 'r') as file:
            for word in file:
                self.vocab[word.split()[0]] = len(self.vocab)
                self.rev_vocab[len(self.vocab) - 1] = word.split()[0]
        self.cls_index = self.vocab['[CLS]']
        self.sep_index = self.vocab['[SEP]']
        self.pad_index = self.vocab['[PAD]']
    def batch_encode(self, seqs):
        code = self.tokenizer(
            seqs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=50,
        )['input_ids']
        return code

    def batch_decode(self, indices):
        sequences = ""
        for j in indices:
            if j == self.cls_index or j == self.pad_index:
                continue
            if j == self.sep_index:
                break
            sequences += self.rev_vocab[int(j)]

        return sequences

    def decode(self, seq):
        s = ""
        for i in seq:
            if i == self.cls_index or i == self.pad_index or i == self.sep_index:
                continue
            # if i == self.sep_index:
            #     break
            s += self.rev_vocab[int(i)]
        return s


def main():
    args = create_argparser().parse_args()
    device = ('cuda' if th.cuda.is_available() else 'cpu')
    print("creating model and diffusion...")
    # myTokenizer = tokenizer(args.vocab_path)
    myTokenizer = PretainedTokenizer()
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

    prior, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    prior.load_state_dict(
        th.load(args.prior_path, map_location="cpu")
    )
    prior.to(device)
    if args.use_fp16:
        prior.convert_to_fp16()
    prior.eval()

    model_emb = th.nn.Embedding(
        num_embeddings=args.vocab_size,
        embedding_dim=args.n_embd,
        _weight=model.get_input_embeddings().weight.clone().cpu()
    ).eval().requires_grad_(False)

    config = ProGenConfig(vocab_size=args.vocab_size,
                          n_positions=args.n_positions,
                          n_ctx=args.n_ctx,
                          n_embd=args.n_embd,
                          n_layer=2)

    text_tokenizer = TextTokenizer()
    text_encoder = TextEncoder().to(device)
    text_encoder.load_state_dict(torch.load(args.text_encoder_path))

    pep_encoder = PepEncoder(config).to(device)
    pep_encoder.load_state_dict(torch.load(args.pep_encoder_path))
    pep_tokenizer = PretainedTokenizer()

    translator = FacModel(config).to(device)
    translator.load_state_dict(torch.load(args.translator_path))
    # pep_tokenizer = PepTokenizer('../mapping/vocab.txt')

    def cond_fn(inputs_embeds, timesteps, reference=None):
        assert reference is not None
        text_ids = reference[1]

        with th.enable_grad():
            x_in = inputs_embeds.detach().requires_grad_(True)
            pep_features = pep_encoder(None, x_in)
            text_features = text_encoder(text_ids)
            logits = text_features @ pep_features.t()
            return th.autograd.grad(logits.sum(), x_in)[0] * args.classifier_scale


    def model_fn(inputs_embeds, timesteps, reference=None):
        assert reference is not None
        # return model(inputs_embeds=inputs_embeds, timesteps=timesteps, self_condition=reference[0])

        return model(inputs_embeds=inputs_embeds, timesteps=timesteps, self_condition=reference[0])

    print("sampling...")
    acc = []
    for length in args.seq_len:
        all_peptides = []
        # ref_data = load_data(myTokenizer, length, 'train', args.cls, args.batch_size)
        # ref_text, ref_pep = get_text_peptides('../dataset/backup/ensemble_test.csv')
        # text_tokens = text_tokenizer.batch_encode(ref_text)
        # pep_tokens = pep_tokenizer.batch_encode(ref_pep)

        # text description for generation
        test_text = ['This is a peptide: target inactive']
        test_tokens = text_tokenizer.batch_encode(test_text)
        test_tokens = test_tokens.repeat(args.batch_size, 1).to(device)

        # sentence = {'src': [], 'trg': []}
        # for i in range(len(test_tokens)):
        #     sentence['src'].append(text_tokens[i])
        #     sentence['trg'].append(pep_tokens[i])
        # ref_data = BatchDataset(sentence)
        # ref_data = DataLoader(ref_data, shuffle=True, batch_size=args.batch_size)
        num = 0

        while len(all_peptides) < args.num_samples:
            model_kwargs = {}
            # text, pep = next(iter(ref_data))
            # text = text.to(device)
            text = test_tokens.to(device)
            text_z = text_encoder(text)
            ref_text = text_z / text_z.norm(dim=-1, keepdim=True)
            # input_ids = pep.to(device)
            # pep_z = pep_encoder(input_ids)
            # pep_z_norm = pep_z / pep_z.norm(dim=-1, keepdim=True)

            timesteps = torch.tensor([0] * args.batch_size, device=device)
            fac_text_z = translator(inputs_embeds=ref_text, timesteps=timesteps)
            fac_text_z_norm = fac_text_z / fac_text_z.norm(dim=-1, keepdim=True)

            condition = fac_text_z_norm
            condition = condition.unsqueeze(1).repeat(1, 50, 1)

            model_kwargs["reference"] = [condition, test_tokens]
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            samples = sample_fn(
                model_fn,
                noise=None,
                shape=(args.batch_size, length, args.n_embd),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                top_p=1,
                clamp_step=0,
                clamp_first=False,
                x_start=None,
                device=device,
                cond_fn=None,
                # denoised_fn=partial(denoised_fn_round, args, model_emb),
            )
            sample = samples[-1]
            sample = model.get_logits(sample)
            _, indices = torch.topk(sample, k=1, dim=-1)
            indices = indices.squeeze(-1)
            all_peptides.extend([s.detach().cpu().numpy() for s in indices])
            print(f"created {args.batch_size} samples")
            torch.cuda.empty_cache()
        with jsonlines.open(os.path.join(args.sample_path, 'samples.jsonl'), 'w') as file:
            for i, seq in enumerate(all_peptides):
                temp = {}
                decoded_sequence = myTokenizer.batch_decode(seq)
                print(decoded_sequence)
        # print(fid)
        print(f"length {length} sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=20,
        max_loop=50,
        use_ddim=False,
        sample_path="../sample/",
        vocab_path="../mapping/vocab.txt",
        prior_path="../checkpoints/w_pretraining_w_inference.pt",
        model_path="../checkpoints/w_pretraining_w_inference.pt",
        pep_encoder_path="../checkpoints/pep_encoder.pt",
        text_encoder_path="../checkpoints/text_encoder.pt",
        translator_path="../checkpoints/translator.pt",
        classifier_scale=1.0,
        embedding_scale=0.0,
        use_fp16=False,
        classifier_use_fp16=False,
        seq_len=[50],
        vocab_size=30,
        class_cond=True,
        cls=2,

    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()
