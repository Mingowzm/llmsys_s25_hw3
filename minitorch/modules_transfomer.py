import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from.tensor_functions import (
    Attn_Softmax,
    LayerNorm
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd    : Dimensionality of embeddings and hidden states
            n_head    : Number of heads
            p_dropout : Dropout ratio for dropout layer
            causal    : If True, then apply a causal mask during self-attention
            bias      : If True, then apply a bias in Linear layers

        Attributes:
            q_projection   : Linear layer projecting input to Q matrix
            k_projection   : Linear layer projecting input to K matrix
            v_project      : Linear layer projecting input to V matrix
            out_projection : Linear output projection layer
            dropout        : Dropout layer
        """
        self.backend   = backend
        self.n_embd    = n_embd
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head
        self.use_fused_kernel = use_fused_kernel

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END YOUR SOLUTION

    def create_causal_mask(self, bs, nh, seq_len):
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        mask = -np.finfo(datatype).max * np.triu(np.ones((bs, nh, seq_len, seq_len), dtype=datatype), 1) # Correct version for Assignment 3.
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """Project x to Q, transpose of K, V for self attention

        Args:
            x: embeddings or hidden states (batch_size x seq_len x n_embd)

        Returns:
            Q   : The Query Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            K^T : The Key Matrix Transposed (batch_size x num_heads x attn_hidden_dim x seq_len)
            V   : The Value Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        x_2d = x.view(batch_size * seq_len, n_embd)
        q_2d = self.q_projection(x_2d)
        k_2d = self.k_projection(x_2d)
        v_2d = self.v_projection(x_2d)
        q = q_2d.view(batch_size, seq_len, n_embd)
        k = k_2d.view(batch_size, seq_len, n_embd)
        v = v_2d.view(batch_size, seq_len, n_embd)
        head_dim = self.attn_hidden_dim
        q = q.view(batch_size, seq_len, self.n_head, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_head, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_head, head_dim).permute(0, 2, 1, 3)
        kT = k.permute(0, 1, 3, 2)
        # raise NotImplementedError
        ### END YOUR SOLUTION
        return q, kT, v

    def self_attention(self, q, kT, v):
        """Given q, kT, and v of sizes defined above, return the result of MultiHeadAttention as described in the writeup
        softmax((q @ kT) / sqrt(attn_hidden_dim)) @ V.
        NOTE: We have added support for Batch Matrix Multiplication with 4 dimensions.
        This means given tensors A of shape (a, b, m, n) and B of shape (a, b, n, p),
        A @ B will be of the shape (a, b, m, p). Take a moment to consider why we need it.

        Args:
            q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            kT : Keys Tensor of shape (batch_size x num_heads x attn_hidden_dim x seq_len)
            v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None

        ### BEGIN YOUR SOLUTION
        logits = q @ kT / (q_dim ** 0.5)

        if self.use_fused_kernel:
            if self.causal:
                mask = self.create_causal_mask(batch_size, num_head, queries_len)
            else:
                mask = logits.zeros(logits.shape)
            attn = Attn_Softmax.apply(logits, mask)
        else:
            if self.causal:
                mask = self.create_causal_mask(batch_size, num_head, queries_len)
                logits = logits + mask
            attn = softmax(logits, dim=3)

        # att = self.dropout(att)
        out_heads = attn @ v
        out_heads = out_heads.permute(0, 2, 1, 3).contiguous()
        out_concat = out_heads.view(batch_size, queries_len, num_head * q_dim)
        result = out_concat
        # raise NotImplementedError
        ### END YOUR SOLUTION

        return result

    def forward(self, x):
        """Computes MultiHeadAttention with causal masking if needed.

        Args:
            x : Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        q, kT, v = self.project_to_query_key_value(x)
        att_out = self.self_attention(q, kT, v)
        att_out = att_out.view(batch_size * seq_len, n_embd)
        out = self.out_projection(att_out)
        out = out.view(batch_size, seq_len, n_embd)
        out = self.dropout(out)
        return out
        # raise NotImplementedError
        ### END YOUR SOLUTION


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """The Feed Forward Module.

        Args:
            n_embd     : in_size of first linear layer and out_size of last linear layer
            middle_dim : out_size of first linear layer and in_size of last linear layer
            p_dropout  : Dropout probability
            bias       : If bias should be applied in linear layers

        Attributes:
            linear_in  : first linear layer
            linear_out : second linear layer
            dropout    : dropout layer
        """
        ### BEGIN YOUR SOLUTION
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END YOUR SOLUTION

    def forward(self, x):
        """A FFN Module in a Pre-LN Transformer with GELU Activation and dropout.

        Args:
            x : Tensor of shape (batch_size x seq_len x n_embd)

        Returns:
            output : Tensor of shape (batch_size x seq_len x n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN YOUR SOLUTION
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        ### END YOUR SOLUTION

        return x


class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args:
            n_embd : Dimensionality of embeddings and hidden states
            n_head : Number of heads for MultiHeadAttention
            p_dropout : Dropout ratio for dropout layer
            ln_eps : A value added for numerical stability in LayerNorm
            bias : If bias should be added in linear layers

        Attributes:
            ln_1 : First LayerNorm1d layer before MultiHeadAttention
            ln_2 : Second LayerNorm1d layer after MultiHeadAttention
            attention : MultiHeadAttention layer
            ff : FeedForward layer
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError

        self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.attention = MultiHeadAttention(n_embd, n_head, causal=True, p_dropout=p_dropout, bias=bias, backend=backend)
        self.ff = FeedForward(n_embd, p_dropout=p_dropout, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        self.use_fused_kernel = use_fused_kernel
        ### END YOUR SOLUTION

    def forward(self, x):
        """The forward function of a Transformer Layer for a Pre-LN Transformer.

        Args:
            x : Hidden state from previous layers with shape (batch_size, seq_len, n_embd)

        Output:
            output: Hidden state after the Transformer Layer with shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION

        res = x
        x_2d = x.view(batch_size * seq_len, n_embd)

        if self.use_fused_kernel:
            # x_norm = LayerNorm.apply(x_2d, self.ln_1.weight, self.ln_1.bias)
            x_norm = LayerNorm.apply(
                x_2d,
                tensor_from_numpy(self.ln1.weights.value.to_numpy(), backend=self.backend, requires_grad=True),
                tensor_from_numpy(self.ln1.bias.value.to_numpy(), backend=self.backend, requires_grad=True)
            )
        else:
            x_norm = self.ln_1(x_2d)

        x = x_norm.view(batch_size, seq_len, n_embd)
        x = self.attention(x)
        x = self.dropout(x) + res

        res = x
        x_2d = x.view(batch_size * seq_len, n_embd)

        if self.use_fused_kernel:
            x_norm = LayerNorm.apply(
                x_2d,
                tensor_from_numpy(self.ln2.weights.value.to_numpy(), backend=self.backend, requires_grad=True),
                tensor_from_numpy(self.ln2.bias.value.to_numpy(), backend=self.backend, requires_grad=True)
            )
        else:
            x_norm = self.ln_2(x_2d)

        x = x_norm.view(batch_size, seq_len, n_embd)
        x = self.ff(x)
        x = self.dropout(x) + res

        return x
        # raise NotImplementedError
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5,
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False
    ):
        super().__init__()
        """A Full Decoder-only Pre-LN Transformer with 4 Transformer Layers.

        Args:
            n_vocab : Vocabulary size defines the number of different tokens that can be represented by the input.
            n_embd  :  Dimensionality of the embeddings and hidden states.
            n_head  : Number of attention heads for each attention layer in the Transformer.
            n_positions : The maximum sequence length that this model might ever be used with.
            p_dropout : The dropout ratio for any dropout layer.
            ln_eps : The epsilon to use in the layer normalization layers.
            bias : If linear layers should include a bias.

        Attributes:
            token_embeddings : Embedding layer for tokens.
            position_embeddings : Embedding layer for token positions.
            t_layer_1 : 1st Transformer Layer.
            t_layer_2 : 2nd Transformer Layer.
            t_layer_3 : 3rd Transformer Layer.
            t_layer_4 : 4th Transformer Layer.
            dropout : Dropout layer before first transformer layer.
            ln : LayerNorm layer after last transformer layer.
            lm_head : Linear layer for projection from (*, n_embd) to (*, n_vocab)
        """
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_vocab             = n_vocab
        self.use_fused_kernel    = use_fused_kernel
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError
        self.token_embeddings    = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        self.t_layer_1           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_2           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_3           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_4           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.dropout             = Dropout(p_dropout)
        self.ln                  = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.lm_head             = Linear(n_embd, n_vocab, bias=bias, backend=backend)
        ### END YOUR SOLUTION

    def forward(self, idx):
        """A Forward pass of a Decoder-only Transformer Language model.
        Args:
            idx: input of shape (batch_size, seq_len)

        Returns:
            logits: logits of shape (batch_size, seq_len, n_vocab)
        """

        batch_size, seq_len = idx.shape

        ### BEGIN SOLUTION
        # raise NotImplementedError
        # Get Token Embeddings of shape (batch_size, seq_len, n_embd)
        token_emb = self.token_embeddings(idx)
        """
        Create Positional Embeddings of shape (1, seq_len, n_embd)
         - First create a tensor of position ids [0, 1, 2, ..., seq_len - 1] of shape (1, seq_len)
         - Pass the position ids through your positional embedding layer
         - Ensure shape is (1, seq_len, n_embd)
        """
        pos_ids = tensor_from_numpy(np.arange(seq_len).reshape(1, seq_len), backend=self.backend, requires_grad=True)
        pos_emb = self.position_embeddings(pos_ids)
        x = token_emb + pos_emb
        x = self.dropout(x)
        # Pass through each transformer Layer
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)
        # Final LayerNorm
        x_2d = x.view(batch_size * seq_len, self.n_embd)

        if self.use_fused_kernel:
            x_norm = LayerNorm.apply(
                x_2d,
                tensor_from_numpy(self.ln.weights.value.to_numpy(), backend=self.backend, requires_grad=True),
                tensor_from_numpy(self.ln.bias.value.to_numpy(), backend=self.backend, requires_grad=True)
            )
        else:
            x_norm = self.ln(x_2d)

        # x = x_norm.view(batch_size, seq_len, self.n_embd)
        out = self.lm_head(x_norm)
        return out.view(batch_size, seq_len, self.n_vocab)
        ### END SOLUTION
