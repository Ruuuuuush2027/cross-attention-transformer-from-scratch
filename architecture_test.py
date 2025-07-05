import torch
from model import build_transformer

transformer = build_transformer(src_vocab_size = 100, tgt_vocab_size = 200, src_seq_len = 10, tgt_seq_len = 20)

# encode
src_test = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
src_mask = torch.rand(1, 1, 10) # broadcast for both self and cross, batch * 1 * src_seq_len

print(f'Input context shape: {src_test.shape}')
encoder_output = transformer.encode(src_test, src_mask)
print(f'Encoder context shape: {encoder_output.shape}')

# decode
tgt_test = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], dtype=torch.long)
print(f'Target initial shape: {tgt_test.shape}')
tgt_self_mask = torch.rand(1, 20, 20) # batch * tgt_seq_len * tgt_seq_len
# src_mask broadcasted to become batch * tgt_seq_len * src_seq_len
decoder_output = transformer.decode(encoder_output, src_mask, tgt_test, tgt_self_mask)
print(f'Decoder output shape: {decoder_output.shape}')
# print(f'TEST: {decoder_output[:, -1].shape}')
# project
tgt_project = transformer.project(decoder_output)
print(f'Final output shape: {tgt_project.shape}')