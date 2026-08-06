import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


# Takes all agents' observations as input, outputs action probability distributions for all agents
class G2ANet(nn.Module):
    def __init__(self, input_shape, args):
        super(G2ANet, self).__init__()

        # Encoding
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)  # Encodes observations for all agents
        self.h = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # Each agent gets a hidden state from its encoded obs to retain memory of prior observations

        # Hard Attention
        # GRU inputs [[h_i, h_1], [h_i, h_2], ... [h_i, h_n]] with [0, ..., 0],
        # outputs [[h_1], [h_2], ..., [h_n]] and [h_n], where h_j encodes the relationship between agent j and agent i.
        # Input shape: (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)
        # For each of batch_size samples, concatenate each agent's hidden state with the other n_agents - 1 agents' hidden states.
        self.hard_bi_GRU = nn.GRU(args.rnn_hidden_dim * 2, args.rnn_hidden_dim, bidirectional=True)
        # Analyzes h_j to produce a weight for agent j relative to agent i.
        # Outputs 2 dimensions; after Gumbel-Softmax, one dimension is selected:
        # 0 means agent j is ignored, 1 means agent j is attended to.
        # Multiplied by 2 because of bidirectional GRU (hidden state dim = 2 * hidden_dim).
        self.hard_encoding = nn.Linear(args.rnn_hidden_dim * 2, 2)

        # Soft Attention
        self.q = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.rnn_hidden_dim, args.attention_dim)

        # Decoding: takes agent i's h_i and aggregated context x_i, outputs action probability distribution
        self.decoding = nn.Linear(args.rnn_hidden_dim + args.attention_dim, args.n_actions)
        self.args = args
        self.input_shape = input_shape

    def forward(self, obs, hidden_state):
        size = obs.shape[0]  # batch_size * n_agents

        # Encode observations
        obs_encoding = f.relu(self.encoding(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        # Pass through each agent's GRU to get hidden state h
        h_out = self.h(obs_encoding, h_in)  # (batch_size * n_agents, rnn_hidden_dim)

        # Hard Attention
        # Note: GRU (not GRUCell) expects input shape (seq_len, batch_size, dim)
        if self.args.hard:
            # Prepare input for hard attention
            h = h_out.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)  # (batch_size, n_agents, rnn_hidden_dim)
            input_hard = []
            for i in range(self.args.n_agents):
                h_i = h[:, i]  # (batch_size, rnn_hidden_dim)
                h_hard_i = []
                for j in range(self.args.n_agents):  # For agent i, concatenate h_i with every other agent's hidden state
                    if j != i:
                        h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))
                # After j loop: h_hard_i is a list of (n_agents - 1) tensors of shape (batch_size, rnn_hidden_dim * 2)
                h_hard_i = torch.stack(h_hard_i, dim=0)
                input_hard.append(h_hard_i)
            # After i loop: input_hard is a list of n_agents tensors of shape (n_agents - 1, batch_size, rnn_hidden_dim * 2)
            input_hard = torch.stack(input_hard, dim=-2)
            # Final shape: (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)
            input_hard = input_hard.view(self.args.n_agents - 1, -1, self.args.rnn_hidden_dim * 2)

            # Bidirectional GRU with 1 layer: first dim = 2 * 1
            h_hard = torch.zeros((2 * 1, size, self.args.rnn_hidden_dim))
            if self.args.cuda:
                h_hard = h_hard.cuda()
            h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)  # (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)
            h_hard = h_hard.permute(1, 0, 2)  # (batch_size * n_agents, n_agents - 1, rnn_hidden_dim * 2)
            h_hard = h_hard.reshape(-1, self.args.rnn_hidden_dim * 2)  # (batch_size * n_agents * (n_agents - 1), rnn_hidden_dim * 2)

            # Compute hard weights: shape (n_agents, batch_size, 1, n_agents - 1)
            # Extra dimension is needed for the weighted sum below
            hard_weights = self.hard_encoding(h_hard)
            hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)
            hard_weights = hard_weights[:, 1].view(-1, self.args.n_agents, 1, self.args.n_agents - 1)
            hard_weights = hard_weights.permute(1, 0, 2, 3)

        else:
            hard_weights = torch.ones((self.args.n_agents, size // self.args.n_agents, 1, self.args.n_agents - 1))
            if self.args.cuda:
                hard_weights = hard_weights.cuda()

        # Soft Attention
        q = self.q(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, attention_dim)
        k = self.k(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, attention_dim)
        x = []
        for i in range(self.args.n_agents):
            q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # Agent i's query: (batch_size, 1, attention_dim)
            k_i = [k[:, j] for j in range(self.args.n_agents) if j != i]  # Keys from all other agents
            v_i = [v[:, j] for j in range(self.args.n_agents) if j != i]  # Values from all other agents

            k_i = torch.stack(k_i, dim=0)  # (n_agents - 1, batch_size, attention_dim)
            k_i = k_i.permute(1, 2, 0)    # (batch_size, attention_dim, n_agents - 1)
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0)

            # (batch_size, 1, attention_dim) x (batch_size, attention_dim, n_agents - 1) = (batch_size, 1, n_agents - 1)
            score = torch.matmul(q_i, k_i)

            # Scale
            scaled_score = score / np.sqrt(self.args.attention_dim)

            # Softmax to get attention weights
            soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size, 1, n_agents - 1)

            # Weighted sum over the n_agents - 1 dimension: output shape (batch_size, attention_dim)
            x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)
            x.append(x_i)

        # Concatenate each agent's hidden state h with its aggregated context x
        x = torch.stack(x, dim=1).reshape(-1, self.args.attention_dim)  # (batch_size * n_agents, attention_dim)
        final_input = torch.cat([h_out, x], dim=-1)
        output = self.decoding(final_input)

        return output, h_out