from overrides import overrides
from pytorch_transformers.modeling_auto import AutoModel
import torch
from allennlp.common.file_utils import cached_path

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from typing import Dict, List


@TokenEmbedder.register("xlm15_anchored")
class Xlm15EmbedderAnchored(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
    """
    def __init__(self, model_name: str, requires_grad: bool, freeze_num_layers: int, aligning_files: Dict[str, str], xnli_tasks: List[str]) -> None:
        super().__init__()

        self.aligning_layer_num = freeze_num_layers # align by last frozen layer


        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

        #for param in self.transformer_model.parameters():
        #    param.requires_grad = requires_grad 
        
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
        

        xnli_langs = [t[4:] for t in xnli_tasks]
        print(xnli_langs)

        for lang in xnli_langs:
            name = 'aligning_%s' % lang

            aligning_matrix = torch.eye(self.output_dim) # default to identity matrix -> no alignment
            if lang in aligning_files and aligning_files[lang] != '':
                print(lang + " will be aligned") 
                aligninig_path = cached_path(aligning_files[lang])
                aligning_matrix = torch.FloatTensor(torch.load(aligninig_path))

            aligning = torch.nn.Linear(self.output_dim, self.output_dim, bias=False)
            aligning.weight = torch.nn.Parameter(aligning_matrix, requires_grad=False)
            self.add_module(name, aligning)
            


    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor, lang: torch.LongTensor) -> torch.Tensor:  # type: ignore
        # pylint: disable=arguments-differ
        lang_name = self.transformer_model.config.id2lang[str(lang[0][0].item())]
        

        try:
            aligning = getattr(self, 'aligning_{}'.format(lang_name))
        except AttributeError:
            aligning = None

        return self.transformer_model(token_ids, langs=lang, aligning=aligning, aligning_layer_num=self.aligning_layer_num)[0]