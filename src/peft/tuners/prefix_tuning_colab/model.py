# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
import torch


class PrefixColabEncoder(torch.nn.Module):
    r"""
    The `torch.nn` model to encode the prefix.

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example:

    ```py
    >>> from peft import PrefixEncoder, PrefixTuningConfig

    >>> config = PrefixTuningConfig(
    ...     peft_type="PREFIX_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_hidden_size=768,
    ... )
    >>> prefix_encoder = PrefixEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The two-layer MLP to transform the prefix embeddings if
          `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (`batch_size`, `num_virtual_tokens`)

    Output shape: (`batch_size`, `num_virtual_tokens`, `2*layers*hidden`)
    """

    def __init__(self, config, base_model_config=None):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        self.num_individual = config.num_individual
        self.num_groups = config.num_groups
        self.num_tokens_individual = config.num_tokens_individual
        self.num_tokens_group = config.num_tokens_group

        # setup colab mechanism
        self.W_q = torch.nn.Linear(token_dim, token_dim)
        self.W_k = torch.nn.Linear(token_dim, token_dim)
        self.W_v = torch.nn.Linear(token_dim, token_dim)

        if base_model_config is not None:
            if hasattr(base_model_config, "num_key_value_heads"):
                num_key_value_heads = base_model_config.num_key_value_heads
            else:
                num_key_value_heads = None

            num_attn_heads = base_model_config.num_attention_heads

        if self.prefix_projection and not config.inference_mode:
            self.individual_embedding = torch.nn.Embedding(self.num_individual * self.num_tokens_individual, token_dim)        
            self.group_embedding = torch.nn.Embedding(self.num_groups * self.num_tokens_group, token_dim)     
            # Use a two-layer MLP to encode the prefix
            if (num_key_value_heads is not None) and (num_key_value_heads != num_attn_heads):   
                self.transform = torch.nn.Sequential(
                    torch.nn.Linear(token_dim, encoder_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(encoder_hidden_size, num_layers * 2 * num_key_value_heads * (token_dim // num_attn_heads)),
                )
            else:
                self.transform = torch.nn.Sequential(
                    torch.nn.Linear(token_dim, encoder_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
                )
        else:
            if (num_key_value_heads is not None) and (num_key_value_heads != num_attn_heads):
                self.individual_embedding = torch.nn.Embedding(self.num_individual * self.num_tokens_individual, num_layers * 2 * num_key_value_heads * (token_dim // num_attn_heads))
                self.group_embedding = torch.nn.Embedding(self.num_groups * self.num_tokens_group, num_layers * 2 * num_key_value_heads * (token_dim // num_attn_heads))
            else:
                self.individual_embedding = torch.nn.Embedding(self.num_individual * self.num_tokens_individual, num_layers * 2 * token_dim)
                self.group_embedding = torch.nn.Embedding(self.num_groups * self.num_tokens_group, num_layers * 2 * token_dim)

    def forward(self, task_ids: torch.Tensor):
        individual_task_id = task_ids[:, 0].unsqueeze(dim=-1)
        group_task_id = task_ids[:, 1].unsqueeze(dim=-1)
        batch_size = individual_task_id.size(0)

        individual_indices = individual_task_id * self.num_tokens_individual + torch.arange(self.num_tokens_individual, device=individual_task_id.device).unsqueeze(0)
        group_indices = group_task_id * self.num_tokens_group + torch.arange(self.num_tokens_group, device=group_task_id.device).unsqueeze(0)

        # get attention weights
        all_individual_embeddings = self.individual_embedding.weight
        all_individual_embeddings = all_individual_embeddings.unsqueeze(0)
       # print(all_individual_embeddings.size())

        self.Q = self.W_q(all_individual_embeddings)
        self.K = self.W_k(all_individual_embeddings)
        self.V = self.W_v(all_individual_embeddings)
        #print(self.Q.size(), self.K.size(), self.V.size())

        attention_output, attention_weights = self.attention()

        #print(f'attention_output: {attention_output.size()}')
        #print(f'attention_weights: {attention_weights.size()}')

        if self.prefix_projection:
            individual_tokens = self.individual_embedding(individual_indices).view(batch_size, self.num_tokens_individual, -1)
            group_tokens = self.group_embedding(group_indices).view(batch_size, self.num_tokens_group, -1)
            print(f'individual_tokens: {individual_tokens.size()}')
            print(f'group_tokens: {group_tokens.size()}')
            print(f'attention_output: {attention_output.size()}')
            combined_tokens = torch.cat((individual_tokens, group_tokens, attention_output), dim=1)
            print(f'combined_tokens: {combined_tokens.size()}')
            past_key_values = self.transform(combined_tokens)
            print(f'past_key_values: {past_key_values.size()}')
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

    def attention(self):

        d_k = self.Q.size(-1)
        scores = torch.matmul(self.Q, self.K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, self.V)

        return output, attention_weights
