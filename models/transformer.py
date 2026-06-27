import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_size, model_dim, num_heads, num_layers, dim_feedforward, output_size, seq_len):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_size, model_dim)  # Embed input pixels
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, model_dim))  # Learnable positional encoding
        
        # Define transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(model_dim, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()  
        # (batch_size, seq_len, model_dim)
        x = self.embedding(x)
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]  
        
        # Transformer encoder expects (seq_len, batch_size, model_dim)
        # (seq_len, batch_size, model_dim)
        x = x.permute(1, 0, 2)
        # (seq_len, batch_size, model_dim)
        x = self.transformer_encoder(x)  
        
        # Use the representation of the last time step for classification
        # (batch_size, model_dim)
        x = x[-1, :, :]
        # (batch_size, output_size)
        x = self.classifier(x)  
        
        return x
    

class Transformer_v2(nn.Module):
    def __init__(self, input_size, model_dim, num_heads, num_layers, dim_feedforward,
                 output_size, seq_len):
        """
        input_size: Length of x_t (e.g., 1083, for x-coordinates of bounding boxes)
        model_dim: Dimensionality of the Transformer embeddings
        num_heads: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dim_feedforward: Dimensionality of the feedforward layers in the Transformer
        seq_len: Maximum number of timesteps to process
        """
        super(Transformer_v2, self).__init__()
        
        self.model_dim = model_dim
        self.input_size = input_size
        #self.output_size = output_size
        self.seq_len = seq_len
        # Positional encoding (learnable for handling temporal order)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        
        # Embed input vector into model_dim
        self.embedding = nn.Linear(input_size, model_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer to map back to input_size (e.g., 183)
        self.output_layer = nn.Linear(model_dim, output_size)

    def forward(self, x_t, t, past_context):
        """
        x_t: (batch_size, input_size) -> Input vector at current timestep t
        past_context: (batch_size, seq_len, model_dim) -> Context from previous timesteps
        "t:": placeholder for timestep t used for RNNs. Note that transformers dont optimize over time
        Returns:
        refined_prediction: (batch_size, input_size) -> Refined bounding box coordinates
        updated_context: (batch_size, seq_len, model_dim) -> Updated context including current timestep
        """
        batch_size = x_t.size(0)
        channel = x_t.shape[1]
        scale = x_t.shape[2]
        x_t = torch.flatten(x_t, 1)
        # Embed the current input (batch_size, model_dim)
        x_t_emb = self.embedding(x_t)
        # Add sequence dimension: (batch_size, 1, model_dim)
        x_t_emb = x_t_emb.unsqueeze(1)  
        # RNNs take as input tuple (None, None) at start as input
        # exception check tuple as transformer only takes None to
        # accomodate pipeline
        if type(past_context) is tuple:
            past_context = past_context[0]
        # Combine with past context
        if past_context is None:
            # Initialize context with only the current timestep
            # (batch_size, 1, model_dim)
            context = x_t_emb
            print("context shape:", context.shape)
        else:
            # Append current timestep to past context
            # (batch_size, seq_len + 1, model_dim)
            context = torch.cat([past_context, x_t_emb], dim=1)
            print("context cat shape:", context.shape)

        # Handle maximum sequence length (sliding window)
        if context.size(1) > self.seq_len:
            # Keep only the last seq_len timesteps
            context = context[:, -self.seq_len:, :]
            print("context slide shape:", context.shape)

        
        # Add positional encoding
        seq_len = context.size(1)
        # (1, seq_len, model_dim)
        pos_enc = self.positional_encoding[:, :seq_len, :]
        # (batch_size, seq_len, model_dim)
        print("res connect c shape:", context.shape)
        print("res connect pos_enc shape:", pos_enc.shape)
        context = context + pos_enc
        print("res connect context shape:", context.shape) 
        
        # Transformer expects input shape: (seq_len, batch_size, model_dim)
        # (seq_len, batch_size, model_dim)
        context = context.permute(1, 0, 2)
        # (seq_len, batch_size, model_dim)
        encoded_context = self.transformer_encoder(context)
        # Back to (batch_size, seq_len, model_dim)  
        encoded_context = encoded_context.permute(1, 0, 2)
        
        # Use the representation of the last timestep to make predictions
        # (batch_size, model_dim)
        current_context = encoded_context[:, -1, :]
        # (batch_size, input_size)       
        refined_prediction = self.output_layer(current_context)  
        refined_prediction = refined_prediction.reshape(batch_size, channel, scale, scale)
        refined_prediction = refined_prediction.unsqueeze(-1)

        return refined_prediction, encoded_context