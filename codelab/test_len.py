from data_generator.vocab import Vocab
from model.model_config import DefaultConfig


model_config = DefaultConfig()
voc_abstr = Vocab(model_config, vocab_path=model_config.path_abstr_voc)