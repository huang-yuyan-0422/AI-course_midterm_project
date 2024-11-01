# AI-course_midterm_project

This is the midterm assignment for the AI Design and Application course. Requirements: Train a language model. Choose your own dataset, model architecture, and training method; there are no restrictions on size or languages.

## Code Foundation

The main code of this project is based on [nanoGPT](https://github.com/karpathy/nanoGPT). On this basis, the datasets have been changed, and the model structure and parameters have been adjusted. The main modifications are as follows:

### Setting Enhanced_tokenizer

The `tokenizer_utils.py` file defines a class named `EnhancedTokenizer`, which uses the tiktoken library to obtain the GPT-2 encoder and adds special character handling to support more complex text.

### Enhancement of Model Structure

In the `model.py` file, the `CausalSelfAttention` class has added relative position encoding, attention temperature parameters, and gating mechanisms to enhance the model's attention mechanism. The `MLP` class uses a combination of GELU and ReLU activation functions, introduces learnable residual connection weights, and adds a scaling factor for residual connections in the `Block` class.

### Parameter Adjustments

After changing the dataset, the parameters have been mainly adjusted to increase `n_layer`, `n_embd`, `learning_rate`, and added dropout to improve the model's capacity and regularization ability.

### Enhancement of Generation Function

In the `generate` method of the `GPT` class, top-k and top-p sampling strategies have been added to support more flexible text generation.

## Run Results

During the training process, the generated training and validation loss graphs are shown below. The `val_loss` decreases as the training steps increase, and no overfitting has occurred.

![val_loss~steps](C:\Users\Cecilia\Desktop\nanoGPT\figures\下载 (1).png)

A portion of the generated text is displayed as follows:


'''
"I wish so much him to go a man, not all you may be be so sometimes.  "

"The doctor wouldn't be likely could be a lifeboat was work that she was no
the face.  It's the widow here for a big as a break for the road, and they
want to be many to her find the first.
---------------
Jessie was a land more in the boat dressmakers.  She was the things like
Miss Perkins said, and it was a lifeboat of one of a possible.
---------------
"I can't be a cottage.  "

"I don't know for the shop of the boat, you must be no many a lot good
more with his woman.  I have not been her sister to be going.  I don't
know what we shall be think the sea of home, sir.  And I shouldn't go to
stay the short, and Jessie had to think it was done.  I'm not been to make
up the storm of business.
---------------


There are still grammatical and spelling errors in the generated text. Future optimization directions include: replacing with a more suitable tokenizer, selecting a larger dataset, and optimizing encoding and decoding strategies (such as punctuation, casing, etc.).
