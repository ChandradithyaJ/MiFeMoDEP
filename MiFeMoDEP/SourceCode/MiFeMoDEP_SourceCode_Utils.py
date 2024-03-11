import ast
import torch
import torch.nn as nn

# preprocess the git diffs to have the code part and the +, - signs
def preprocess_diff(diff_content):    
    lines = str(diff_content).split("\n")
    removed_lines = []
    added_lines = []
    for line in lines:
        if line.startswith("@@"):
            continue  # Skip header lines
    
        if line.startswith("+"):
            line = line[line.index('+')+1:]
            if line.startswith("#"):
                continue
            hashtag_idx = line.find("#")
            if hashtag_idx > 0:
                line = line[:hashtag_idx]
            added_lines.append(line)
        elif line.startswith("-"):
            line = line[line.index('-')+1:]
            if line.startswith("#"):
                continue
            hashtag_idx = line.find("#")
            if hashtag_idx > 0:
                line = line[:hashtag_idx]
            removed_lines.append(line)

    added_code = ' \n '.join(list(set(added_lines)))
    removed_code = ' \n '.join(list(set(removed_lines)))
    return added_code + " " + removed_code


string_to_number = {}
counter = 0

def assign_number(string):
    global counter
    global string_to_number
    # Check if the string is already in the dictionary
    if string in string_to_number:
        return string_to_number[string]

    # If not, assign a new number and update the dictionary
    else:
        counter += 1
        string_to_number[string] = counter
        return counter

def parse_code(code_string):
    tree = ast.parse(code_string)
    return tree

node_type_map = {
    ast.Assign: "assignment",
    ast.Name: "variable",
    ast.Call: "function call",
}

def build_pdg(node, edges, parent=None):
    # Get node type and label
    node_type = node_type_map.get(type(node), "other")
    label = str(node)  # Replace with your desired label extraction logic

    # Add node to PDG
    pdg_node = {"type": node_type, "label": label}

    # Handle different node types and create edges
    if isinstance(node, ast.Assign):
        for target in node.targets:
              edges.append((assign_number(str(target)), assign_number(pdg_node["label"])))

    # Recursively traverse child nodes
    for child in ast.iter_child_nodes(node):
        build_pdg(child, edges, pdg_node)

    return pdg_node

def extract_pdg(code_string):
    tree = parse_code(code_string)
    if not tree:
        return None
    edges = []
    pdg_root = build_pdg(tree, edges)
    return {"nodes": [pdg_root], "edges": edges}

def extract_pdg_for_gcn(code_string):
    pdg_data = extract_pdg(code_string)
    if not pdg_data:
        return None
    # Convert edges to PyTorch tensor (edge_index format)
    edge_index = torch.tensor(list(zip(*pdg_data["edges"]))).T

    return edge_index


def preprocess_source_code_for_d2v(code):
    # Define operators and comment delimiters
    operators = "+-*/%=&|^<>!?:."
    comment_start_single = "#"
    comment_start_multi = "'''"

    # Remove single-line comments and strip whitespace after removal
    cleaned_lines = []
    for line in code.splitlines():
        cleaned_line = line.split(comment_start_single)[0].strip()
        cleaned_lines.append(cleaned_line)

    # Join cleaned lines while preserving newlines
    cleaned_code = "\n".join(cleaned_lines)

    # Remove multi-line comments
    while comment_start_multi in cleaned_code:
        start_index = cleaned_code.find(comment_start_multi)
        end_index = cleaned_code.find(comment_start_multi, start_index + 3)
        cleaned_code = cleaned_code[:start_index] + cleaned_code[end_index + 3:]

    # Remove operators
    cleaned_code = "".join([char for char in cleaned_code if char not in operators])
    # Remove blank lines
    cleaned_code = "\n".join([line for line in cleaned_code.splitlines() if line])
    
    return cleaned_code.split("\n")


# CodeBERT
max_len = 10000
def get_CodeBERT_context_embeddings(tokenizer, model, source_code):
    tokens = tokenizer.tokenize(source_code)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # if there are more than L tokens give them in batches
    L = 512 # token stream size
    N0 = 64 # overlap factor for context preservation

    tensor_embeddings = []
    if len(tokens_ids) > L:
        i = 0
        while i + N0 < len(tokens_ids):
            tokens_batch = tokens_ids[i : i + L]
            with torch.no_grad():
                embeddings = model(torch.tensor(tokens_batch).to(torch.int64)[None,:])[0]
            tensor_embeddings.append(embeddings)
            i += (L - N0)
    else:
        with torch.no_grad():
            tensor_embeddings = model(torch.tensor(tokens_ids).to(torch.int64)[None,:])[0]
        
    # concatenate
    if len(tokens_ids) > 512:
        tensor_embeddings = torch.concat(tensor_embeddings, dim=1)
        
    if tensor_embeddings.size(1) > max_len:
        tensor_embeddings = tensor_embeddings[:, :max_len, :]
    else:
        padding = torch.zeros((1, max_len - tensor_embeddings.size(1), 768))
        tensor_embeddings = torch.cat((tensor_embeddings, padding), dim=1)
    
    tensor_embeddings = torch.squeeze(tensor_embeddings, dim=0)
    return tensor_embeddings


class Encoder(nn.Module):
    def __init__(self, hidden_features, output_features):
        super(Encoder, self).__init__()
        self.ff1 = nn.Linear(200, hidden_features)
        self.ff2 = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        if type(x) == tuple and type(x[0]) == str:
            x = ast.literal_eval(x)
        x = self.ff1(x)
        x = nn.ReLU()(x)
        x = self.ff2(x)
        x = nn.Softmax()(x)
        return x
    
hidden_features = 128
output_features = 768  # Number of output features (encoded representation)
enc_model = Encoder(hidden_features, output_features)


def get_PDG_embeddings(d2v_model, source_code):
    
    d2v_rep = d2v_model.infer_vector(preprocess_source_code_for_d2v(source_code))
    d2v_rep = torch.from_numpy(d2v_rep)
    
    edge_index = extract_pdg_for_gcn(source_code)
    edge_index = edge_index.reshape(-1)
    if(edge_index.size(0) != 0):
        edge_index = edge_index/torch.max(edge_index)
        
    if edge_index.size(0) > 50:
        edge_index = edge_index[:50]
    else:
        edge_index = nn.functional.pad(edge_index, (0, 50 - edge_index.size(0)), value=0)
    
    enc_inp = torch.cat((d2v_rep, edge_index), dim=0)

    #return enc_inp.unsqueeze(dim=0)
    return enc_inp

def encode_PDG(enc_inp):
    embeds = enc_model(enc_inp)
    return embeds

# Dataset
class DefectorsTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_point = self.dataframe.iloc[idx]
        cb_embeds = data_point['cb_embeds']
        pdg_enc_inp = data_point['pdg_enc_inp']
        label = data_point['target']
        
        input_tensor1 = cb_embeds
        input_tensor2 = pdg_enc_inp
        label_tensor = torch.tensor(label)

        return input_tensor1, input_tensor2, label_tensor
    
class DefectorsTorchDatasetEncoderTraining(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_point = self.dataframe.iloc[idx]
        cb_embeds = data_point['cb_embeds']
        pdg_enc_inp = data_point['pdg_enc_inp']
        label = data_point['target']
        
        input_tensor = pdg_enc_inp
        label_tensor = torch.tensor(label)

        return input_tensor, label_tensor
    
    
class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()

        self.l1 = nn.Linear(768, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.l3 = nn.Linear(64, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pdg_enc_inp):
        # get embeddings
        pdg_embeds = encode_PDG(pdg_enc_inp)
        #embeds = torch.concat((cb_embeds, pdg_embeds), dim=1)
        embeds = pdg_embeds
        
        z = self.l1(embeds)
        z = self.leakyrelu(z)
        z = self.bn1(z)
        z = self.l2(z)
        z = self.leakyrelu(z)
        z = self.bn2(z)
        z = self.l3(z)
        predictions = self.sigmoid(z)
        
        return predictions