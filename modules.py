from __future__ import annotations
from typing import Callable, Optional, Union

import math
import torch

Activation = Callable[[torch.Tensor], torch.Tensor]
Device = Union[str, torch.device]


def masked_softmax(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Compute softmax of a tensor using a mask.
    Parameters
    ----------
    x
        Argument of softmax.
    mask
        Mask tensor.
    dim
        Dimension along which softmax will be computed.
    Returns
    -------
    torch.Tensor
        Masked softmax of `x`.
    """
    X = (mask * x).masked_fill(mask == 0, -float("inf"))
    return torch.softmax(X, dim)


def dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute scaled dot product attention.
    Parameters
    ----------
    query
        Query vectors.
        Shape: :math:`(N_{queries},d_q)`
    key
        Key vectors.
        Shape: :math:`(N_{keys},d_q)`
    mask
        Mask tensor to ignore query-key pairs, by default None.
        Shape: :math:`(N_{queries},N_{keys})`
    Returns
    -------
    torch.Tensor
        Scaled dot product attention between each query and key vector.
        Shape: :math:`(N_{queries},N_{keys})`
    """

    _, d = key.shape

    pre_attn = query @ key.transpose(0, 1) / math.sqrt(d)

    return masked_softmax(pre_attn, mask, dim=1)


def aggregate_mask(
    groups: torch.Tensor,
    num_groups: int,
    num_items: int,
    mean: bool = False,
    device: Device = "cpu",
) -> torch.Tensor:
    """Create a mask to aggregate items into groups.
    Parameters
    ----------
    groups
        Tensor of group indices.
    num_groups
        Number of groups.
    num_items
        Number of items.
    mean
        If True, mask aggregates by averaging rather than summing, by default False.
    device
        Device on which mask will be created, by default "cpu".
    Returns
    -------
    torch.Tensor
        Mask to aggregate items into groups.
    """
    M = torch.zeros(num_groups, num_items, device=device)
    M[groups, torch.arange(num_items)] = 1

    if mean:
        M = torch.nn.functional.normalize(M, p=1, dim=1)

    return M


def nodewise_mask(
    edge_index: torch.Tensor,
    mean: bool = False,
    device: Device = "cpu",
) -> torch.Tensor:
    """Create a mask for nodewise aggregation of incoming graph edges.
    Parameters
    ----------
    edge_index
        Edge indices.
        Shape: :math:`(2,N_{edges})`
    mean
        If True, mask aggregates by averaging rather than summing, by default False.
    device
        Device on which mask will be created, by default "cpu".
    Returns
    -------
    torch.Tensor
        Mask for nodewise edge aggregation.
        Shape: :math:`(N_{nodes},N_{edges})`
    """
    _, N_e = edge_index.shape
    s, r = edge_index
    return aggregate_mask(r, r.max() + 1, N_e, mean=mean, device=device)


def batchwise_mask(
    batch: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    mean: bool = False,
    device: Device = "cpu",
) -> torch.Tensor:
    """Create a mask for batchwise aggregation of graph nodes or edges.
    Parameters
    ----------
    batch
        Tensor of batch indices.
        Shape: :math:`(N_{nodes},)`
    edge_index
        Tensor of edge indices, by default None.
        Shape: :math:`(2,N_{edges})`
    mean
        If True, mask aggregates by averaging rather than summing, by default False.
    device
        Device on which mask will be created, by default "cpu".
    Returns
    -------
    torch.Tensor
        Mask for batchwise aggregation.
        Shape: :math:`(N_{batch},N_{nodes})` if `edge_index = None`
    """
    if edge_index is not None:
        # masking for edge aggregation
        _, N_e = edge_index.shape
        s, r = edge_index
        return aggregate_mask(
            batch[r], batch[r].max() + 1, N_e, mean=mean, device=device
        )
    else:
        # masking for node aggregation
        N_v = len(batch)
        return aggregate_mask(batch, batch.max() + 1, N_v, mean=mean, device=device)


def batchwise_edge_mean(
    edges: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    device="cpu",
) -> torch.Tensor:
    M = batchwise_mask(batch, edge_index, mean=True, device=device)

    return M @ edges


def batchwise_node_mean(
    nodes: torch.Tensor, batch: torch.Tensor, device="cpu"
) -> torch.Tensor:
    M = batchwise_mask(batch, mean=True, device=device)

    return M @ nodes


class Dense(torch.nn.Module):
    def __init__(
        self,
        *features: int,
        bias: bool = True,
        activation: Optional[Activation] = None,
    ) -> None:
        """Dense feed-forward neural network.
        Parameters
        ----------
        *features
            Number of features at each layer.
        bias
            If False, each layer will not learn an additive bias; by default True.
        activation
            Activation function.
        """
        super().__init__()

        layers = []

        for n_in, n_out in zip(features, features[1:]):
            layers.append(torch.nn.Linear(n_in, n_out, bias=bias))
            if activation is not None:
                layers.append(activation)

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.
        Parameters
        ----------
        x
            Input tensor.
            Shape: :math:`(N, *, H_{in})`
        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N, *, H_{out})`
        """
        return self.seq(x)


class Scale(torch.nn.Module):
    def __init__(self, shift: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.
        Parameters
        ----------
        x
            Input tensor.
        Returns
        -------
        torch.Tensor
            Shifted and scaled output tensor.
        """
        return (x - self.shift) / self.scale


class Transform(torch.nn.Module):
    def __init__(self, **modules) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d = {}
        for k, t in data.items():
            if k in self.modules:
                d[k] = self.modules[k](t)
            else:
                d[k] = t

        return d


class GraphNetwork(torch.nn.Module):
    """Graph Network from https://arxiv.org/abs/1806.01261."""

    def __init__(
        self,
        edge_model: Optional[torch.nn.Module] = None,
        node_model: Optional[torch.nn.Module] = None,
        global_model: Optional[torch.nn.Module] = None,
        num_layers: Optional[int] = 1,
    ) -> None:
        """[summary]
        Parameters
        ----------
        edge_model
            Edge update network, by default None.
        node_model
            Node update network, by default None.
        global_model
            Global update network, by default None.
        num_layers
            Number of passes, by default 1.
        """
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        self.num_layers = num_layers

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute forward pass.
        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        edges
            Edge features, by default None.
            Shape: :math:`(N_{edges},d_e)`
        u
            Global features, by default None.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor, by default None.
            Shape: :math:`(N_{nodes},)`
        Returns
        -------
        torch.Tensor
            Output node feature tensor.
            Shape: :math:`(N_{nodes},d_v)`
        torch.Tensor
            Output edge index tensor.
            Shape: :math:`(2,N_{edges})`
        torch.Tensor
            Output edge feature tensor.
            Shape: :math:`(N_{edges},d_e)`
        torch.Tensor
            Output global feature tensor.
            Shape: :math:`(N_{batch},d_u)`
        torch.Tensor
            Output batch tensor.
            Shape: :math:`(N_{nodes},)`
        """
        if batch is None:
            N_v, *_ = nodes.shape
            batch = torch.zeros((N_v,), dtype=torch.long)

        for _ in range(self.num_layers):
            if self.edge_model is not None:
                edges = self.edge_model(nodes, edge_index, edges, u, batch).reshape(
                    edges.shape
                )

            if self.node_model is not None:
                nodes = self.node_model(nodes, edge_index, edges, u, batch).reshape(
                    nodes.shape
                )

            if self.global_model is not None:
                u = self.global_model(nodes, edge_index, edges, u, batch).reshape(
                    u.shape
                )

        return nodes, edge_index, edges, u, batch


class UpdateBlock(torch.nn.Module):
    def __init__(self, d_x: int, d_m: int, message_depth: int = 1) -> None:
        """Update block.

        Parameters
        ----------
        d_x
            Input dimension.
        d_m
            Message dimension.
        message_depth
            Number of hidden layers used to compute message from input, by default 1.
        """
        super().__init__()

        message_dims = [d_m] * message_depth
        self.m = torch.nn.Sequential(
            Dense(d_x, *message_dims, activation=torch.nn.SELU()),
            torch.nn.Linear(d_m, d_m),
        )

        self.gru = torch.nn.GRUCell(d_m, d_m)

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            Shape: :math:`(N_{batch},d_x)`
        h0 : torch.Tensor
            Initial hidden state tensor.
            Shape: :math:`(N_{batch},d_x)`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{batch},d_m)`
        """
        m = self.m(x)
        return self.gru(m, h0)


class KeyValueAttention(torch.nn.Module):
    def __init__(
        self, d_Q: int, d_K: int, d_attn: int, num_heads: int, d_out: int
    ) -> None:
        """Attentive aggregation block.

        Parameters
        ----------
        d_Q
            Query vector dimension.
        d_K
            Key vector dimension.
        d_attn
            Latent space dimension.
        num_heads
            Number of attention heads.
        d_out
            Output dimension.
        """
        super().__init__()

        self.queries = torch.nn.ModuleList(
            [Dense(d_Q, d_attn, bias=False) for _ in range(num_heads)]
        )
        self.keys = torch.nn.ModuleList(
            [Dense(d_K, d_attn, bias=False) for _ in range(num_heads)]
        )
        self.values = torch.nn.ModuleList(
            [Dense(d_K, d_attn, bias=False) for _ in range(num_heads)]
        )
        self.gates = torch.nn.ModuleList(
            [Dense(d_Q, 1, activation=torch.nn.Sigmoid()) for _ in range(num_heads)]
        )

        self.out = Dense(d_attn, d_out)

    def forward(
        self, x_Q: torch.Tensor, x_K: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x_Q
            Query vector.
            Shape: :math:`(N_{batch},d_Q)`
        x_K
            Key vector.
            Shape: :math:`(N_{batch},d_K)`
        mask
            Mask tensor.
            Shape: :math:`(N_{nodes},N_{clusters})`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{batch},d_{out})`
        """

        gates = torch.stack([G(x_Q).squeeze(-1) for G in self.gates])
        heads = []

        for query, key, value in zip(self.queries, self.keys, self.values):
            Q = query(x_Q)
            K = key(x_K)
            V = value(x_K)

            attn = dot_product_attention(Q, K, mask)

            heads.append(attn @ V)

        heads = torch.stack(heads)

        return self.out(torch.einsum("ijk, ij -> jk", heads, gates))


class EdgeModel(torch.nn.Module):
    def __init__(self, d_v, d_e, d_u, message_depth=1):
        super().__init__()

        self.block = UpdateBlock(2 * d_v + d_u, d_e, message_depth)

    def forward(self, nodes, edge_index, edges, u, batch):

        s, r = edge_index
        x = torch.cat([nodes[r], nodes[s], u[batch][s]], dim=1)

        return edges + torch.nn.SELU()(self.block(x, edges))


class NodeModel(torch.nn.Module):
    def __init__(
        self,
        d_v,
        d_e,
        d_u,
        num_heads,
        d_attn,
        message_depth=1,
    ):
        super().__init__()

        self.edge_attn = KeyValueAttention(
            d_v + d_u, d_e + d_v + d_u, d_attn, num_heads, d_e
        )

        self.block = UpdateBlock(d_e + d_u, d_v, message_depth)

    def forward(self, nodes, edge_index, edges, u, batch):

        s, r = edge_index
        x_Q = torch.cat([nodes, u[batch]], dim=1)
        x_K = torch.cat([edges, nodes[r], u[batch][r]], dim=1)

        mask = nodewise_mask(edge_index, device=edges.device)

        agg_edges = self.edge_attn(x_Q, x_K, mask)

        x = torch.cat([agg_edges, u[batch]], dim=1)

        return nodes + torch.nn.SELU()(self.block(x, nodes))


class GraphModel(torch.nn.Module):
    def __init__(
        self,
        d_v,
        d_e,
        d_u,
        num_heads,
        d_attn,
        message_depth=1,
    ):
        super().__init__()

        self.edge_attn = KeyValueAttention(d_u, d_e + d_u, d_attn, num_heads, d_e)

        self.node_attn = KeyValueAttention(d_u, d_v + d_u, d_attn, num_heads, d_v)

        self.block = UpdateBlock(d_e + d_v, d_u, message_depth)

    def forward(self, nodes, edge_index, edges, u, batch):

        s, r = edge_index
        x_Q = torch.cat([u], dim=1)
        x_K = torch.cat([edges, u[batch][r]], dim=1)

        mask = batchwise_mask(batch, edge_index, device=edges.device)
        agg_edge = self.edge_attn(x_Q, x_K, mask)

        x_Q = torch.cat([u], dim=1)
        x_K = torch.cat([nodes, u[batch]], dim=1)

        mask = batchwise_mask(batch, device=edges.device)
        agg_node = self.node_attn(x_Q, x_K, mask)

        x = torch.cat([agg_edge, agg_node], dim=1)

        return u + torch.nn.SELU()(self.block(x, u))


class Embed(torch.nn.Module):
    def __init__(self, d_v: int, d_e: int, d_u: int) -> None:
        """Embed initial graph features.

        Parameters
        ----------
        d_v
            Node feature dimension.
        d_e
            Edge feature dimension.
        d_u
            Graph feature dimension.
        """
        super().__init__()

        self.node_embedding = Dense(d_v, d_v, d_v, activation=torch.nn.SELU())
        self.edge_embedding = Dense(d_e, d_e, d_e, activation=torch.nn.SELU())
        self.graph_embedding = Dense(d_u, d_u, d_u, activation=torch.nn.SELU())

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edges: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        edges
            Edge features, by default None.
            Shape: :math:`(N_{edges},d_e)`
        u
            Global features, by default None.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor, by default None.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Output node feature tensor.
            Shape: :math:`(N_{nodes},d_v)`
        torch.Tensor
            Output edge index tensor.
            Shape: :math:`(2,N_{edges})`
        torch.Tensor
            Output edge feature tensor.
            Shape: :math:`(N_{edges},d_e)`
        torch.Tensor
            Output global feature tensor.
            Shape: :math:`(N_{batch},d_u)`
        torch.Tensor
            Output batch tensor.
            Shape: :math:`(N_{nodes},)`
        """
        nodes = torch.cat([nodes, self.node_embedding(nodes)], dim=1)
        edges = torch.cat([edges, self.edge_embedding(edges)], dim=1)
        u = torch.cat([u, self.graph_embedding(u)], dim=1)

        return nodes, edge_index, edges, u, batch


class Net(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        m_e: int,
        K_v: int,
        d_attn_v: int,
        m_v: int,
        K_u: int,
        d_attn_u: int,
        m_u: int,
    ) -> None:
        super().__init__()
        d_v = 10
        d_e = 9
        d_u = 3
        d_y = 20

        self.embed = Embed(d_v, d_e, d_u)

        d_v *= 2
        d_e *= 2
        d_u *= 2

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            edge_model = EdgeModel(d_v, d_e, d_u, m_e)
            node_model = NodeModel(d_v, d_e, d_u, K_v, d_attn_v, m_v)
            graph_model = GraphModel(d_v, d_e, d_u, K_u, d_attn_u, m_u)
            self.layers.append(GraphNetwork(edge_model, node_model, graph_model))

        self.agg_nodes = KeyValueAttention(d_u, d_v, 100, 5, d_v)
        self.agg_edges = KeyValueAttention(d_u, d_e, 100, 5, d_e)

        self.energy_readout = Dense(
            d_v + d_e + d_u, d_v + d_e + d_u, d_y, activation=torch.nn.ReLU()
        )
        self.strength_readout = Dense(
            d_v + d_e + d_u, d_v + d_e + d_u, d_y, activation=torch.nn.ReLU()
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edges: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        edges
            Edge features, by default None.
            Shape: :math:`(N_{edges},d_e)`
        u
            Global features, by default None.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor, by default None.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Output node feature tensor.
            Shape: :math:`(N_{nodes},d_v)`
        torch.Tensor
            Output edge index tensor.
            Shape: :math:`(2,N_{edges})`
        torch.Tensor
            Output edge feature tensor.
            Shape: :math:`(N_{edges},d_e)`
        torch.Tensor
            Output global feature tensor.
            Shape: :math:`(N_{batch},d_u)`
        torch.Tensor
            Output batch tensor.
            Shape: :math:`(N_{nodes},)`
        """
        nodes, edge_index, edges, u, batch = self.embed(
            nodes, edge_index, edges, u, batch
        )
        for gn in self.layers:

            nodes, edge_index, edges, u, batch = gn(nodes, edge_index, edges, u, batch)

        agg_node = batchwise_node_mean(nodes, batch, device=nodes.device)
        agg_edge = batchwise_edge_mean(edges, edge_index, batch, device=nodes.device)

        x = torch.cat([agg_node, agg_edge, u], dim=1)

        energies = torch.cumsum(self.energy_readout(x).view(-1, 20), dim=1)
        strengths = self.strength_readout(x).view(-1, 20)

        return torch.stack([energies, strengths], dim=-1)
