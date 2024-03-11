import re, os
import torch
import torch.nn as nn
import more_itertools
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

char_to_remove = ['+','-','*','/','=','++','--','\\','<str>','<char>','|','&','!']

def is_comment_line(code_line, comments_list):
    '''
        input
            code_line (string): source code in a line
            comments_list (list): a list that contains every comments
        output
            boolean value
    '''

    code_line = code_line.strip()

    if len(code_line) == 0:
        return False
    elif code_line.startswith('#'):
        return True
    elif code_line in comments_list:
        return True

    return False

def is_empty_line(code_line):
    '''
        input
            code_line (string)
        output
            boolean value
    '''

    if len(code_line.strip()) == 0:
        return True

    return False

def preprocess_code_line(code_line):
    '''
        input
            code_line (string)
    '''

    code_line = re.sub("\'\'", "\'", code_line)
    code_line = re.sub("\".*?\"", "<str>", code_line)
    code_line = re.sub("\'.*?\'", "<char>", code_line)
    code_line = re.sub('\b\d+\b','',code_line)
    code_line = re.sub("\\[.*?\\]", '', code_line)
    code_line = re.sub("[\\.|,|:|;|{|}|(|)]", ' ', code_line)

    for char in char_to_remove:
        code_line = code_line.replace(char,' ')

    code_line = code_line.strip()

    return code_line

def preprocess_code(code_str):
    '''
        input
            code_str (multi line str)
    '''
    if(code_str is None):
        return ''
    code_str = code_str.decode("latin-1")
    code_lines = code_str.splitlines()

    preprocess_code_lines = []
    is_comments = []

    # multi-line comments
    comments = re.findall(r'("""(.*?)""")|(\'\'\'(.*?)\'\'\')', code_str, re.DOTALL)
    comments_temp = []
    for tup in comments:
        temp = ''
        for s in tup:
            temp += s
        comments_temp.append(temp)
    comments_str = '\n'.join(comments_temp)
    comments_list = comments_str.split('\n')

    for l in code_lines:
        l = l.strip()
        is_comment = is_comment_line(l,comments_list)
        is_comments.append(is_comment)

        if not is_comment:
            l = preprocess_code_line(l)

        preprocess_code_lines.append(l)

    return ' \n '.join(preprocess_code_lines)


max_seq_len = 50
def prepare_code2d(code_list):
    '''
        input
            code_list (list): list that contains code each line (in str format)
        output
            code2d (nested list): a list that contains list of tokens with padding by '<pad>'
    '''
    # content to list(content)
    code_list = str(code_list)
    code_list = code_list.splitlines()
    code2d = []

    for c in code_list:
        c = re.sub('\\s+',' ',c)

        c = c.lower()

        token_list = c.strip().split()
        total_tokens = len(token_list)

        token_list = token_list[:max_seq_len]

        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>']*(max_seq_len-total_tokens)

        code2d.append(token_list)

    return code2d


def get_code3d_and_label(df):
    '''
        input
            df (DataFrame): a dataframe from get_df()
        output
            code3d (nested list): a list of code2d from prepare_code2d()
            all_file_label (list): a list of file-level label
    '''
    code_3d = prepare_code2d(df['content'].to_numpy())
    all_file_label = df['target'].to_numpy().tolist()
    return code_3d, all_file_label


def train_word2vec_model(embedding_dim = 50, df=None):

    w2v_path = './word2vec'

    save_path = w2v_path+'/'+'w2v'+str(embedding_dim)+'dim.bin'

    if os.path.exists(save_path):
        print('word2vec model at {} is already exists'.format(save_path))
        return

    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)

    train_code_3d, _ = get_code3d_and_label(df)

    all_texts = list(more_itertools.collapse(train_code_3d[:],levels=2))

    word2vec = Word2Vec(all_texts,vector_size=embedding_dim, min_count=1,sorted_vocab=1)

    word2vec.save(save_path)
    print('save word2vec model at path {} done'.format(save_path))
    return word2vec

def get_x_vec(code_3d, word2vec):
    x_vec = [[[
        word2vec.wv.key_to_index[token] if token in word2vec.wv.key_to_index else len(word2vec.wv.key_to_index)
        for token in text
    ] for text in texts] for texts in code_3d]
    return x_vec


def pad_code(code_list_3d, max_sent_len, max_seq_len, limit_sent_len=True, mode='train'):
  padded = []

  for file in code_list_3d:
    sent_list = []
    for line in file:
      new_line = line
      # Truncate if line is longer than max_seq_len
      if len(line) > max_seq_len:
        new_line = line[:max_seq_len]
          #edited here ..just trying
      elif len(line) <= max_seq_len:
          new_line = line + [0] * (max_seq_len - len(new_line))
      sent_list.append(new_line)

    # Pad the entire file (all sentences) to max_sent_len with zeros
    padded_file = sent_list + [[0] * max_seq_len for _ in range(max_sent_len - len(sent_list))]

    # If in training mode and `limit_sent_len` is True, keep only the first max_sent_len sentences
    if mode == 'train' and limit_sent_len:
        padded_file = padded_file[:max_sent_len]
      
    padded.append(padded_file)
    # print(padded_file)
  return padded


def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.vectors)
    # add zero vector for unknown tokens
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1,embed_dim)))
    return word2vec_weights

def get_dataloader(code_vec, label_list, batch_size, max_sent_len):
  y_tensor = torch.FloatTensor([label for label in label_list])
  
  # Ensure padding happens to max_sent_len
  code_vec_pad = pad_code(code_vec, max_sent_len,max_seq_len)
  
  # Print shapes for debugging (optional)
  print(f"code_vec shape: {len(code_vec)}")
  print(f"Y_tensor shape: {len(y_tensor)}")
  print(f"code_vec_pad shape: {len(code_vec_pad)}")
  
  tensor_dataset = TensorDataset(torch.tensor(code_vec_pad), y_tensor)
  dl = DataLoader(tensor_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
  return dl

class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        """
        vocab_size: number of words in the vocabulary of the model
        embed_dim: dimension of word embeddings
        word_gru_hidden_dim: dimension of word-level GRU; biGRU output is double this size
        sent_gru_hidden_dim: dimension of sentence-level GRU; biGRU output is double this size
        word_gru_num_layers: number of layers in word-level GRU
        sent_gru_num_layers: number of layers in sentence-level GRU
        word_att_dim: dimension of word-level attention layer
        sent_att_dim: dimension of sentence-level attention layer
        use_layer_norm: whether to use layer normalization
        dropout: dropout rate; 0 to not use dropout
        """
        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout)

        self.fc = nn.Linear(2 * sent_gru_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, code_tensor):
        
        code_lengths = []
        sent_lengths = []

        for file in code_tensor:
            code_line = []
            code_lengths.append(len(file))
            for line in file:
                code_line.append(len(line))
            sent_lengths.append(code_line)
        
        code_tensor = code_tensor.type(torch.LongTensor)
        code_lengths = torch.tensor(code_lengths).type(torch.LongTensor)
        sent_lengths = torch.tensor(sent_lengths).type(torch.LongTensor)
        
        code_embeds, word_att_weights, sent_att_weights, sents = self.sent_attention(code_tensor, code_lengths, sent_lengths)

        scores = self.fc(code_embeds)
        final_scrs = self.sig(scores)

        return final_scrs, word_att_weights, sent_att_weights, sents

class SentenceAttention(nn.Module):
    """
    Sentence-level attention module. Contains a word-level attention module.
    """
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, word_gru_num_layers, word_att_dim, use_layer_norm, dropout)

        # Bidirectional sentence-level GRU
        self.gru = nn.GRU(2 * word_gru_hidden_dim, sent_gru_hidden_dim, num_layers=sent_gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * sent_gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Sentence-level attention
        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, sent_att_dim)

        # Sentence context vector u_s to take dot product with
        # This is equivalent to taking that dot product (Eq.10 in the paper),
        # as u_s is the linear layer's 1D parameter vector here
        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, code_tensor, code_lengths, sent_lengths):

        # Sort code_tensor by decreasing order in length
        code_lengths, code_perm_idx = code_lengths.sort(dim=0, descending=True)
        code_tensor = code_tensor[code_perm_idx]
        sent_lengths = sent_lengths[code_perm_idx]

        # Make a long batch of sentences by removing pad-sentences
        # i.e. `code_tensor` was of size (num_code_tensor, padded_code_lengths, padded_sent_length)
        # -> `packed_sents.data` is now of size (num_sents, padded_sent_length)
        packed_sents = pack_padded_sequence(code_tensor, lengths=code_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_sents.batch_sizes

        # Make a long batch of sentence lengths by removing pad-sentences
        # i.e. `sent_lengths` was of size (num_code_tensor, padded_code_lengths)
        # -> `packed_sent_lengths.data` is now of size (num_sents)
        packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=code_lengths.tolist(), batch_first=True)

    
    
        # Word attention module
        sents, word_att_weights = self.word_attention(packed_sents.data, packed_sent_lengths.data)

        sents = self.dropout(sents)

        # Sentence-level GRU over sentence embeddings
        packed_sents, _ = self.gru(PackedSequence(sents, valid_bsz))

        if self.use_layer_norm:
            normed_sents = self.layer_norm(packed_sents.data)
        else:
            normed_sents = packed_sents

        # Sentence attention
        att = torch.tanh(self.sent_attention(normed_sents))
        att = self.sentence_context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val)

        # Restore as documents by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as documents by repadding
        code_tensor, _ = pad_packed_sequence(packed_sents, batch_first=True)

        # Compute document vectors
        code_tensor = code_tensor * sent_att_weights.unsqueeze(2)
        code_tensor = code_tensor.sum(dim=1)

        # Restore as documents by repadding
        word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, valid_bsz), batch_first=True)

        # Restore the original order of documents (undo the first sorting)
        _, code_tensor_unperm_idx = code_perm_idx.sort(dim=0, descending=False)
        code_tensor = code_tensor[code_tensor_unperm_idx]

        word_att_weights = word_att_weights[code_tensor_unperm_idx]
        sent_att_weights = sent_att_weights[code_tensor_unperm_idx]

        return code_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # output (batch, hidden_size)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        embeddings: embeddings to init with
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        """
        self.embeddings.weight.requires_grad = freeze

    def forward(self, sents, sent_lengths):
        """
        sents: encoded sentence-level data; LongTensor (num_sents, pad_len, embed_dim)
        return: sentence embeddings, attention weights of words
        """
        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]

        sents = self.embeddings(sents)

        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_words.batch_sizes

        # Apply word-level GRU over word embeddings
        packed_words, _ = self.gru(packed_words)

        if self.use_layer_norm:
            normed_words = self.layer_norm(packed_words.data)
        else:
            normed_words = packed_words

        # Word Attenton
        att = torch.tanh(self.attention(normed_words.data))
        att = self.context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val) # att.size: (n_words)

        # Restore as sentences by repadding
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        # Compute sentence vectors
        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx]

        att_weights = att_weights[sent_unperm_idx]

        return sents, att_weights
    
weight_dict = {}
def get_loss_weight(labels):
    '''
        input
            labels: a PyTorch tensor that contains labels
        output
            weight_tensor: a PyTorch tensor that contains weight of defect/clean class
    '''
    label_list = labels.cpu().numpy().squeeze().tolist()
    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])

    weight_tensor = torch.tensor(weight_list).reshape(-1,1)
    return weight_tensor