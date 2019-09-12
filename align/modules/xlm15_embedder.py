from overrides import overrides
from pytorch_transformers.modeling_auto import AutoModel
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("xlm15")
class Xlm15Embedder(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
    """
    def __init__(self, model_name: str, requires_grad: bool = True, freeze_num_layers: int = 0) -> None:
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

        for param in self.transformer_model.parameters():
            param.requires_grad = requires_grad 
        
        def get_layer_num(n):
            parts = n.split(".")
            for part in parts:
                try:
                    return int(part)
                except ValueError:
                    continue
            return 0

        if freeze_num_layers > 0:
            for n, p in self.transformer_model.named_parameters():
                layer_num = get_layer_num(n)
                if layer_num < freeze_num_layers:
                    p.requires_grad = False


    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor, lang: torch.LongTensor = None) -> torch.Tensor:  # type: ignore
        # pylint: disable=arguments-differ
        return self.transformer_model(token_ids, langs=lang)[0]