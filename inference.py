import torch
import streamlit as st
import pandas as pd
device = "cpu"
from data_utils import data_prep
from model_utils import model


misc_data = pd.read_pickle("misc_data.pkl")
model_path = misc_data["model_path"]
vocab_size = misc_data["vocab_size"]

# Load the DataLoader and model
dl = data_prep.DataLoader()
dl.load()
blm = model.BigramLM(dl.vocab_size)
blm.load_state_dict(torch.load(model_path))
blm = blm.to(device)
blm.eval()  # Set the model to evaluation mode


# App framework
st.title("Generate text with LM")
prompt = st.text_input("Enter any character")
if prompt:
    input_idx = dl.encode(prompt)
    input_tensor = torch.tensor(input_idx, device=device)
    input_tensor = input_tensor.reshape(1, 1)
    st.write(dl.decode(blm.generate(input_tensor, 64)[0].tolist()))


