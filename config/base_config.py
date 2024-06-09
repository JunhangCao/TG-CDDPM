import argparse


def create_argparser():
    defaults = dict(
        data_dir="../dataset",
        val_data_dir="../dataset",
        vocab_path='../mapping/vocab.txt',
        checkpoint_dir='../checkpoints',
        timesteps=500,
        train_epoches=300,
        lr=1e-4,
        weight_decay=0.0,
        batch_size=32,
        seq_len=[i for i in range(20, 51)],
        is_need_classifier=False,
        label_num=2,

        n_positions=256,
        n_ctx=256,
        n_embd=256,
        n_layer=2,

    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")