from dataclasses import dataclass, field

from transformers import Wav2Vec2Config


@dataclass
class WV(Wav2Vec2Config):

    def __init__(self, **kwargs):
        super().__init__(hidden_size=hidden_size,
                         num_hidden_layers=num_hidden_layers,
                         num_attention_heads=num_attention_heads,
                         intermediate_size=
                         conv_dim=(4, 4),
                         conv_stride=(1, 1),
                         conv_kernel=(3, 3),
                         num_conv_pos_embeddings=3,
                         num_conv_pos_embedding_groups=1,
                         proj_codevector_dim=4,
                         classifier_proj_size=3,
                         num_labels=2, **kwargs)


c = WV()
print(c.num_labels, c.vocab_size)
