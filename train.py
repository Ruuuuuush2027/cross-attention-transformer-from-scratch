import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from typing import Dict
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config: Dict, ds, lang):
    # config['tokenizer_file'] = "../tokenizers/tokeinzer_{}"
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    tokenizer = None
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(iterator = get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config: Dict):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train', cache_dir = 'data') # keep in data dir, clean

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print('max len src:', max_len_src)
    print('max len tgt:', max_len_tgt)

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # initialize decoder input
    decoder_input = torch.tensor([sos_idx], dtype = source.dtype).to(device)
    next_word = torch.tensor([eos_idx + 1])
    while decoder_input.size(1) < max_len and not next_word.item() == eos_idx:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)

        out = model.decode(source, source_mask, decoder_input, decoder_mask) # (1, len_decoder_input, d_model), next word embed
        prob = model.project(out[:, -1]) # (1, d_model), [:, -1] only performs on the last two dimension, extracts last vector of every batch

        _, next_word = torch.max(prob, dim = 1) # go through dim 1 and return index
        
        decoder_input = torch.cat([decoder_input, next_word.type_as(source).to(device)], dim = 1)

    return decoder_input.squeeze()

def validate_model(model, val_loader, tokenizer_tgt, max_len, device, print_msg, writer, num_examples = 2):
    model.eval()

    source_texts = []
    expected = []
    predicted = []

    with torch.inference_mode():
        for id, batch in enumerate(val_loader):
            if id == num_examples:
                break
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)

            model_out = greedy_decode(model, encoder_output, encoder_mask, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # function given by tqdm so not disrupt progress bar
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
    
    if writer:
        pass # possibly do some metric, etc.

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    num_classes = tokenizer_tgt.get_vocab_size() # number of output classes

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1)

    initial_epoch = 0

    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location = torch.device(device))
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
    
    print("[INFO]: training started")
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'processing epoch {epoch:02d}')
        for id, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            y = batch['label'].to(device) # (B, seq_len)

            # run tensors through transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            y_pred = model.project(decoder_output) # (B, seq_len, num_classes)
            
            # y_pred -> (B*seq_len, num_classes); y -> (B*seq_len); what cross entropy wants
            loss = loss_fn(y_pred.view(-1, num_classes), y.view(-1))

            # log on p bar and tensor board
            batch_iterator.set_postfix({"loss": f'{loss.item():6.3f}'})
            writer.add_scalar('train loss', loss.item(), epoch)
            writer.flush()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model_filename = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_filename)

        validate_model(model, val_dataloader, tokenizer_tgt, config['seq_len'], device,
                       batch_iterator.write, writer)

if __name__ == '__main__':
    config = get_config()
    train_model(config)



# train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(get_config())
# print(next(iter(train_dataloader))['encoder_mask'].shape)
# print(next(iter(train_dataloader))['decoder_mask'].shape)