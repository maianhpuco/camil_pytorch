import torch
import torch.nn as nn

from src.custom_layers import NeighborAggregator, LastSigmoid, MILAttentionLayer, CustomAttention
from src.nystromformer import NystromAttention  

class CAMIL(nn.Module):
    def __init__(self, args):
        """
        Build the architecture of the Graph Attention Network (Graph Att Net)
        Parameters:
        ----------
        args: Argument class containing all the necessary model parameters
        """
        super(CAMIL, self).__init__()

        self.input_shape = args.input_shape
        self.args = args
        self.n_classes = args.n_classes
        self.subtyping = args.subtyping

        # Initialize layers and components of the model
        self.attcls = MILAttentionLayer(
            input_dim=512, 
            weight_params_dim=128, 
            use_gated=False, 
            )
        # self.wv = nn.Linear(512, 512)
        self.neigh = NeighborAggregator(output_dim=1)
        self.nyst_att = NystromAttention(
            dim=512, 
            dim_head=64, 
            heads=8, 
            num_landmarks=256, 
            pinv_iterations=6
            )
        self.encoder = Encoder()

        # Define the final fully connected layer for classification
        if self.subtyping:
            self.class_fc = LastSigmoid(
                input_dim=512,output_dim=self.n_classes, subtyping=self.subtyping, pooling_mode='sum')
        else:
            self.class_fc = LastSigmoid(
                input_dim=512, output_dim=1, subtyping=self.subtyping, pooling_mode='sum')


    def forward(self, bag, adjacency_matrix):
        """
        Define the forward pass of the model
        """
        # Pass through the encoder
        xo, alpha = self.encoder([bag, adjacency_matrix])
        
        # Slice and print parts of the output for debugging
        # print("- xo (full):", xo.shape)
        # print("- xo (first 5):", xo[:5])  # Slicing to print the first 5 elements
        # print("- alpha (full):", alpha.shape)
        # print("- alpha (first 5):", alpha[:5])  # Slicing to print the first 5 elements

        # Attention mechanism
        k_alpha = self.attcls(xo)
        
        # print("- k_alpha (full):", k_alpha.shape)
        # print("- k_alpha (first 5):", k_alpha[:5])  # Slicing to print the first 5 elements
        # print("---- xo * k_alpha")
        
        # Apply attention to the encoder output
        attn_output = torch.mul(k_alpha, xo)
        
        # print("- attention output shape:", attn_output.shape)
        # print("- attention output (first 5):", attn_output[:5])  # Slicing to print the first 5 elements
        
        # Final classification layer
        out = self.class_fc(attn_output)
        
        # print("- out (full):", out.shape)
        # print("- out (first 5):", out[:5])  # Slicing to print the first 5 elements
        return out, alpha, k_alpha
    
class Encoder(nn.Module):
    """
    Encoder module that processes input data through multiple layers, including Nystrom attention, custom attention, and a neighbor aggregator. 
    The encoder generates an output with enhanced feature representations by aggregating information from neighbors in the graph structure 
    and applying attention mechanisms.

    Args:
        nn.Module: Inherits from PyTorch's nn.Module to define a custom neural network layer.
    Returns:
        tuple: 
            - torch.Tensor: The final encoded output after applying attention mechanisms and transformations.
            - torch.Tensor: The attention scores computed during the forward pass, representing the importance of each instance or feature. 
    Attributes:
        custom_att (CustomAttention): A custom attention mechanism that computes the attention matrix.
        wv (nn.Linear): A fully connected layer for transforming dense feature vectors into the appropriate output dimensionality.
        neigh (NeighborAggregator): A layer that aggregates neighborhood information from the graph structure.
        nyst_att (NystromAttention): A Nystrom-based attention mechanism that approximates the self-attention computation efficiently. 

         Forward Pass:
        1. Applies Nystrom attention on the dense input to capture global dependencies in the data.
        2. Adds the residual connection between the Nystrom attention output and the original dense input.
        3. Computes the attention matrix using a custom attention mechanism.
        4. Aggregates information from the neighbors of each node using the NeighborAggregator layer.
        5. Transforms the dense input using a fully connected layer (wv).
        6. Computes attention weights by element-wise multiplying the normalized attention scores and transformed values.
        7. Adjusts the encoded output based on the attention weights. 
    """ 
    def __init__(
        self, 
        embedding_dim=512, weight_params_dim=256):
        super(Encoder, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.weight_params_dim = weight_params_dim 
        # Initialize components like custom attention, neighbor aggregator, and Nystrom Attention
        self.custom_att = CustomAttention(
            input_dim=self.embedding_dim, 
            weight_params_dim=self.weight_params_dim
            ) 
        self.wv = nn.Linear(self.embedding_dim, self.embedding_dim)  # Linear layer for value transformation
        
        self.neigh = NeighborAggregator(output_dim=1)
        
        # The NystromAttention is assumed to be implemented elsewhere in PyTorch
        self.nyst_att = NystromAttention(
            dim=self.embedding_dim, dim_head=64, heads=8, num_landmarks=16, pinv_iterations=6) #256 

    def forward(self, inputs):
        dense = inputs[0]  # Dense input data (e.g., feature vectors)
        sparse_adj = inputs[1]  # Sparse adjacency matrix for graph structure
        # Apply the Nystrom attention on the dense input
        encoder_output = self.nyst_att(dense.unsqueeze(0))  # Add an extra dimension to the input (batch dimension) 
        xg = encoder_output.squeeze(0)  # Remove the extra dimension (batch dimension)
        # print("- xg:", xg.shape)
        
        # Combine the encoder output with the original dense input (residual connection)
        encoder_output = xg + dense
        # print("- encoder_output: ", encoder_output.shape)    
        # Compute the attention matrix and neighbor aggregation
        attention_matrix = self.custom_att(encoder_output)
        # print("- attention matrix Q * K: ", attention_matrix.shape)
        
        # print("--- Compute w_i, input includes the attention_matrix and spare_adj")
        # print("+ sparse_adj:", sparse_adj.shape)  
        norm_alpha, alpha = self.neigh([attention_matrix, sparse_adj])
        
        
        # print("- norm_alpha:", norm_alpha.shape)
        # print("- alpha: ", alpha.shape)
        # print("-- norm_alpha: ", norm_alpha[:10])
        
        # Transform the dense input using the value transformation layer (wv)
        value = self.wv(dense)
        # print("- value", value.shape)
        
        # print("-----") 
        # print("+ norm_alpha:", norm_alpha.shape)
        # print("+ value:", value.shape)
        # Compute element-wise multiplication of attention scores and transformed values
        norm_alpha = norm_alpha.unsqueeze(1)  # Reshape to [300, 1]
        norm_alpha = norm_alpha.expand(-1, value.size(1))  # Expand to [300, 512] 
        xl = norm_alpha * value
        # print("- xl:", xl.shape)
        # Calculate the attention weights and adjust the encoded output
        # print("-------sigmod(l)")
        wei = torch.sigmoid(-xl)  # Sigmoid activation for weighting
        # print("- wei", wei.shape)
        squared_wei = wei ** 2
        # print("-----This code is m=sigmoid(l)")
        # print(wei[:5, :2])
        # print(wei[:5, :2])
        xo = (xl * 2 * squared_wei) + 2 * encoder_output * (1 - squared_wei)
        # print("- xo", xo.shape)
        # print("- alpha", alpha.shape)
        # This line calculates the final output xo 
        # by combining the weighted attention values (xl) 
        # with the residual connection (encoder_output). 
        # The result depends on the attention weights (wei and squared_wei). 
        return xo, alpha
    