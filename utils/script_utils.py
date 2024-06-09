import argparse
import inspect

import model.gaussian_diffusion as gd
from model.backend import PepClassifier, ProGenForCausalLM
from config.backen_config import ProGenConfig
from model.respace import SpacedDiffusion, space_timesteps

NUM_CLASSES = 2


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=500,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        vocab_size=30,
        n_positions=256,
        n_ctx=256,
        n_embd=256,
        n_layer=6,
        n_fea=56,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        vocab_size=30,
        n_positions=256,
        n_ctx=256,
        n_embd=256,
        n_layer=6,
        n_fea=56,
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    vocab_size,
    n_positions,
    n_ctx,
    n_embd,
    n_layer,
    n_fea,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    model = create_model(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_fea=n_fea,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    vocab_size,
    n_positions,
    n_ctx,
    n_embd,
    n_layer,
    n_fea,
):
    config = ProGenConfig(vocab_size=vocab_size, n_positions=n_positions, n_ctx=n_ctx,
                          n_embd=n_embd, n_layer=n_layer, n_fea=n_fea)
    return ProGenForCausalLM(
        config
    )


def create_classifier_and_diffusion(
    vocab_size,
    n_positions,
    n_ctx,
    n_embd,
    n_layer,
    n_fea,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    classifier = create_classifier(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_ctx,
        n_layer=n_layer,
        n_embd=n_embd,
        n_fea=n_fea,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    vocab_size,
    n_positions,
    n_ctx,
    n_embd,
    n_layer,
    n_fea,
):

    config = ProGenConfig(vocab_size=vocab_size,n_positions=n_positions,
                       n_ctx=n_ctx,n_embd=n_embd,n_layer=n_layer,n_fea=n_fea)

    # return PepDiscriminator(config)
    return PepClassifier(config)
    # return CNNClassifier(n_embd, 2, 8, 4, 0.2)

def create_gaussian_diffusion(
    *,
    steps=500,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=True,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas=learn_sigma,
        sigma_small=sigma_small,
        use_kl=use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )
    # SpacedDiffusion(
    #     use_timesteps=space_timesteps(steps, timestep_respacing),
    #     betas=betas,
    #     model_mean_type=(
    #         gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
    #     ),
    #     model_var_type=(
    #         (
    #             gd.ModelVarType.FIXED_LARGE
    #             if not sigma_small
    #             else gd.ModelVarType.FIXED_SMALL
    #         )
    #         if not learn_sigma
    #         else gd.ModelVarType.LEARNED_RANGE
    #     ),
    #     loss_type=loss_type,
    #     rescale_timesteps=rescale_timesteps,
    # )
    return diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


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
