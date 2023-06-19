import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # creating an loopup -logprob table to get next token -logprob
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # doing forward propagation
    def forward(seld, idx, targets):

        # idx and targets are both (B, T) tensor of intergers
        logits = seld.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # Doing some shape correction which pytoch expects
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    

    # generate text like GPT
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) 
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(idx, None)
            # focus only on the last time stamp
            logits = logits[:, -1, :] #becomes (B, C)
            # apply softmax to get probabilites
            probs = F.softmax(logits, dim=-1) #(B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            # append sampled iindex to the running sequence
            idx = torch.cat([idx, idx_next], dim=1) #(B, T+1)
        return idx