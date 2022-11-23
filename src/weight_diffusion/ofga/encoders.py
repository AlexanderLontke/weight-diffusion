from transformers import BertTokenizer


def get_pretrained_bert_tokenizer(**kwargs):
    return BertTokenizer.from_pretrained(**kwargs)
