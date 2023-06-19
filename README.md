Bigram Language Model - Text Generation (Insipred from Andrej Karpathy tutorial)
This project aims to create a Bigram Language Model that generates text similar to how GPT generates text but at very small scale. The model operates at the character level and uses a small text dataset called "input.txt" for training.

Overview
The Bigram Language Model is a statistical language model that predicts the probability of the next character based on the previous character. It analyzes the frequency of character pairs (bigrams) in the training data and uses this information to generate text Though we have taken a different path of training pytorch nn model to converge in the same idea.

Installation
To run the Bigram Language Model, follow these steps:

Clone the repository:
git clone https://github.com/arindamcodes/bigramlm.git
Change into the project directory:
cd bigramlm

Install the necessary dependencies. This project requires Python 3 and the following packages:

You can install the required packages by running:
pip install -r requirements.txt

Usage
To generate text using the Bigram Language Model, follow these steps:

Open the terminal and navigate to the project directory.

Run the following command to execute the script:

To do training,
python trainer.py
The script will load the training data, build the Bigram Language Model and will create model weights

To do infernce,
streamlit run inference.py
The script will load the training data, load the pretrained model weights and take input a single character input in browser and the generated text will be displayed in the browser

Example Input
'a'


Example Ouput:
"atherdeit s, fe my: HASAnal to lit; iselledou st.

Whooo oromeatr"


