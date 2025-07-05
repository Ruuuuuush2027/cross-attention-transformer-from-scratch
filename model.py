import torch
from torch import nn
import math

# embedding
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x): 
        # x: BATCH * SEQ_LEN, [[1, 50, 30], [7, 21, 31]] ... each array just a series of numbers each number a token
        # to BATCH * SEQ_LEN * d_model
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    # not learned
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # create matrix of (seq_len, d_model), will be overwrite later
        pe = torch.zeros(seq_len, d_model)

        # vector of (seq_len, 1), for each d_model embedding
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) 
        # within each d_model, even position use cosine, odd position use sine
        # for every even term within a d_embedding, 1 / 10000 ^ (2i / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sine to even, cosine to odd; odd 2i+1 position uses pos as 2i
        # (seq_len, 1) * (1, d_model/2) = (seq_len, d_model/2), apply sine and cosine 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension, (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # tensor kept not as a parameter but saved, and save as self.pe; if only self.pe, no save on state_dict
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # broadcast, no need to expand
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added
    
    def forward(self, x: torch.Tensor): # x: B * SEQ * D_MODEL
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    # after attention produces a mix of contexts, B * SEQ * D_MODEL
    # each D_MODEL is now a mix of contexts
    # FFN then remembers what can be infered from each context, which feeds to next
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        res = self.linear_1(x)
        res = self.relu(res)
        res = self.dropout(res)
        return self.linear_2(res)
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        # make sure d_model divisible by h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def _attention(queries, keys, vals, mask, dropout: nn.Dropout):
        d_k = queries.shape[-1]

        attention_scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dim() == 3:  # (batch, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len) where all 1 is broadcasted
            # replace all values which mask==0 is true with -1e9
            attention_scores.masked_fill_(mask == 0, -1e9) 
        
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ vals), attention_scores # attention scores for visualization
    
    def forward(self, q, k, v, mask):
        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        val = self.w_v(v)

        # split across heads, (B, seq_len, d_model) -> (B, seq_len, h, d_k) -> (B, h, seq_len, d_k)
        queries = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        keys = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        vals = val.view(val.shape[0], val.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock._attention(queries, keys, vals, mask, self.dropout)
        # (B, h, seq_len, d_k) -> (B, seq_len, h, d_k) -> (B, seq_len, h * d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.msa = self_attention_block # msa = multihead self attention
        self.ffb = feed_forward_block
        self.res_connection = nn.ModuleList([ResidualConnection(dropout), ResidualConnection(dropout)])
    
    def forward(self, x, src_mask):
        x = self.res_connection[0](x, lambda x: self.msa(x, x, x, src_mask)) # labmda function to allow calling
        x = self.res_connection[1](x, self.ffb)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.msa = self_attention_block
        self.mca = cross_attention_block # mca = multihead cross attention
        self.ffb = feed_forward_block
        # three of them this time
        self.res_connection = nn.ModuleList([ResidualConnection(dropout), ResidualConnection(dropout), ResidualConnection(dropout)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # tgt_mask is for decoder mask, self_attention
        # src_mask is for cross attention
        x = self.res_connection[0](x, lambda x: self.msa(x, x, x, tgt_mask))
        x = self.res_connection[1](x, lambda x: self.mca(x, encoder_output, encoder_output, src_mask))
        x = self.res_connection[2](x, self.ffb)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask): # output B*seq_len*d_model
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model:int, vocab_size: int): # vocab_size: num of words in a corpus
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (B, seq_len, d_model) -> (B, seq_len, vocab_size)
        # if inference want to take last word, logits[:, -1, :] for each batch
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask): # (B*seq_src) -> (B*seq_src*d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask, tgt, tgt_mask): # (B*seq_src*d_model) and (B*seq_tgt) -> (B*seq_tgt*d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x): # (B*seq_tgt*d_model) -> (B*seq_tgt*tgt_vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Args:
        d_model: number of embedding dim
        N: number of encoder layers in a encoder, same for decoder
        h: number of heads for each attention block
        d_ff: number of hidden params in each Feed Forward Block
        tgt_seq_len: max threshold as tgt_embed and tgt_pos have the positional embed calculation, but it's fine to be less, as long as mask is has same dim
    """
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # encoders and decoders
    encoder_blocks = []
    for _ in range(N):
        encoder_msa = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_ffn = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_msa, encoder_ffn, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_msa = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_mca = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_ffn = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_msa, decoder_mca, decoder_ffn, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize parameters, xavier_uniform_ to speed up learning
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer