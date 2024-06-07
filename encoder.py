import torch
import math
from torch import nn
import torch.nn.functional as F
     
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    print(f"scaled.size() : {scaled.size()}")
    if mask is not None:
        print(f"-- ADDING MASK of shape {mask.size()} --") 
        # Broadcasting add. So just the last N dimensions need to match
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_sequence_length):
            super().__init__()
            self.max_sequence_length = max_sequence_length
            self.d_model = d_model

        def forward(self):
            even_i = torch.arange(0, self.d_model, 2).float()
            denominator = torch.pow(10000, even_i/self.d_model)
            position = (torch.arange(self.max_sequence_length)
                            .reshape(self.max_sequence_length, 1))
            even_PE = torch.sin(position / denominator)
            odd_PE = torch.cos(position / denominator)
            stacked = torch.stack([even_PE, odd_PE], dim=2)
            PE = torch.flatten(stacked, start_dim=1, end_dim=2)
            return PE

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std
        print(f"y: {y.size()}")
        out = self.gamma * y  + self.beta
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f"out: {out.size()}")
        return out

  
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        print(f"x after first linear layer: {x.size()}")
        x = self.relu(x)
        print(f"x after activation: {x.size()}")
        x = self.dropout(x)
        print(f"x after dropout: {x.size()}")
        x = self.linear2(x)
        print(f"x after 2nd linear layer: {x.size()}")
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x
        print("------- ATTENTION 1 ------")
        x = self.attention(x, mask=None)
        print("------- DROPOUT 1 ------")
        x = self.dropout1(x)
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        x = self.norm1(x + residual_x)
        residual_x = x
        print("------- ATTENTION 2 ------")
        x = self.ffn(x)
        print("------- DROPOUT 2 ------")
        x = self.dropout2(x)
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        x = self.norm2(x + residual_x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                     for _ in range(num_layers)])
        self.input_embedding = Input_embedding(d_model,15,drop_prob)

        self.output = MyMLP(d_model)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.layers(x)
        x = self.output(x)
        return x
    


class MyMLP(nn.Module):
    def __init__(self,d_model,drop_prob=0.1):
        super(MyMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(961 * d_model, 1)  # Beispielhafte erste Schicht
        self.fc2 = nn.Linear(1024, d_model)       # Beispielhafte zweite Schicht
        self.fc3 = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)         # Ausgabe-Schicht für (4, 1)
    
    def forward(self, x):
        x = self.flatten(x)  # Transformiere (4, 31, 512) zu (4, 31*512)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)    
        # # x = F.relu(self.fc1(x))
        # # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # x = self.dropout(x)      # Ausgabe ist (4, 1)
        return x


class Input_embedding(nn.Module):

    def __init__(self, d_model, input_dim, drop_prob=0.1 ):
        super(Input_embedding, self).__init__()
        self.linear1 = nn.Linear(input_dim, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.position_encoder = PositionalEncoding(d_model,961)

    def forward(self, x):
        x = self.linear1(x)
        print(f"x after first linear layer: {x.size()}")
        x = self.relu(x)
        print(f"x after activation: {x.size()}")
        x = self.dropout(x)
        print(f"x after dropout: {x.size()}")
        x = self.linear2(x)
        print(f"x after 2nd linear layer: {x.size()}")
        pos = self.position_encoder()
        x = self.dropout(x+pos)
        return x
# d_model = 512
# num_heads = 8
# drop_prob = 0.1
# batch_size = 1
# max_sequence_length = 961
# ffn_hidden = 2048
# num_layers = 5   
# encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
# x = torch.randn( (batch_size, 31, 15) ) # includes positional encoding
# out = encoder(x)
# print(out)