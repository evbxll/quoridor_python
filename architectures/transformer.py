import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
  def __init__(self, tokenizer, d_model, num_heads, num_layers, dim_feedforward, max_len=512, dropout=0.1):
    super(SimpleTransformer, self).__init__()

    self.tokenizer = tokenizer
    vocab_size = len(tokenizer)

    self.num_heads = num_heads

    # Embedding layers for tokens and positions
    self.token_embedding = nn.Embedding(vocab_size, d_model)
    self.position_embedding = nn.Embedding(max_len, d_model)

    # Transformer Encoder using nn.Transformer with only encoder layers
    self.transformer = nn.Transformer(
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=0,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation='relu',
        batch_first=True
    )

    self.attention_weight = nn.Linear(d_model, 1)

    # Output projection layer
    self.output_layer = nn.Linear(d_model, 1)

    self.sig = nn.Sigmoid()

  def forward(self, x):
      # Create positional embeddings
      b, seq_len = x.shape

      position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
      x = self.token_embedding(x) + self.position_embedding(position_ids)
      x = self.transformer.encoder(x)

      # Attention pooling to focus on relevant board positions
      # [b, seq_len, 1]
      attention_weights = torch.softmax(self.attention_weight(x), dim=1)

      # [b, d_model]
      x = (x * attention_weights).sum(dim=1)

      # Final output layer
      out = self.output_layer(x)
      return self.sig(out)
