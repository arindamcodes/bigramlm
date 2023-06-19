import torch

class DataLoader(object):

    # Constructor init
    def __init__(self):
        self.text = None
        self.chars = None
        self.vocab_size = None
        self.encode = None
        self.decode = None
        self.data = None
        self.batch_size = 32
        self.block_size = 8


    # set random seed
    def set_seed(self):
        torch.manual_seed(1337)

    # Read the text data
    def read_data(self, filename='data_utils/input.txt'):
        with open(filename, 'r', encoding='utf-8') as f:
            self.text = f.read()


    # Create vocab or code book from text
    def create_vocab(self):
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)


    # Creation of encoder and decoder
    def create_encode_decode(self):
        stoi = { ch:i for i, ch in enumerate(self.chars)}
        itos = { i:ch for i, ch in enumerate(self.chars)}

        # Endocing the text into numbers
        # the way we tokenize text
        self.encode = lambda s: [stoi[c] for c in s]
        # Decoding the numbers into text
        self.decode = lambda l: ''.join([itos[i] for i in l])

    
    # Create tensor dataset using torch
    def create_tensor_data(self):
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)

    
    ## Create train and test split
    def train_test_split(self): 
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    
    #load the data
    def load(self):
        # batch of x and y
        self.read_data()
        self.create_vocab()
        self.create_encode_decode()
        self.create_tensor_data()
        self.train_test_split()


    # Creating batches from data
    def get_single_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i+self.block_size] for i in ix])
        y = torch.stack([data[i+1: i+self.block_size+1] for i in ix])
        return x, y


    





      

    



















