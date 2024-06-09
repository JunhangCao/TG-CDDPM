from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ProGenConfig(PretrainedConfig):
    model_type = "progen"

    def __init__(
            self,
            vocab_size=50400,
            n_positions=2048,
            n_ctx=2048,
            n_embd=4096,
            n_layer=28,
            n_head=16,
            rotary_dim=64,
            n_inner=None,
            n_fea=None,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_weights=True,
            gradient_checkpointing=False,
            use_cache=True,
            self_condition=True,
            bos_token_id=50256,
            eos_token_id=50256,
            logits_mode=1,
            cls=2,
            **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.n_fea = n_fea
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.self_condition = self_condition
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.logits_mode = logits_mode
        self.cls = cls

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer