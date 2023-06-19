import torch
import pickle
from data_utils import data_prep
from model_utils import model



# hyper params
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
# device = "mps" if torch.backends.mps.is_available() \
#     else "gpu" if torch.cuda.is_available() else "cpu"
device = "gpu" if torch.cuda.is_available() else "cpu"
eval_iters = 200



# Loading the data
dl = data_prep.DataLoader()
dl.load()


# Load model
blm = model.BigramLM(dl.vocab_size)
blm = blm.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(blm.parameters(), lr=learning_rate)


# Monitor over or under fitting
@torch.no_grad()
def estimate_loss(lm):
    out = {}
    lm.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dl.get_single_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = lm(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    lm.train()
    return out


for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(blm)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = dl.get_single_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    # evaluate the loss
    logits, loss = blm.forward(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# Save the model
model_path = "blm.pth"
torch.save(blm.state_dict(), model_path)

dump_dict = {
    "model_path": model_path,
    "vocab_size": dl.vocab_size,
}

with open("misc_data.pkl", "wb") as f:
    pickle.dump(dump_dict, f)

# generate from the model
start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(dl.decode(blm.generate(start_idx, 64)[0].tolist()))