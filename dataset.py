import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat( # sos + src_sentence + eos + pad
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        decoder_input = torch.cat( # sos + sentence + pad
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        label = torch.cat([ # sentence + eos + pad
            torch.tensor(dec_input_tokens, dtype = torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype = torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
             # (1, 1, seq_len), ignore padding in self-attention; when broadcasted it's 1, xxx, seq_len
            "encoder_mask": (encoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int(), 
            # (1, 1, seq_len) & (1, seq_len, seq_len); so only valid 1 on both cases should pass
            "decoder_mask": (decoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), 
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text 
        }
    
def causal_mask(size):
    """
    Produces self attention mask, lower half is one and upper half is zero so filled with -1e99
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int) # everything above diagonal is retained 1; diagonal itself is zero
    return mask == 0 # everything flip, as lower should be one, upper should be zero, filled with -1e99