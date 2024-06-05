# Text-Generator-Model-By-LLM

## Introduction

This repository provides a simple yet comprehensive example of using the GPT-2 language model for text generation. GPT-2, developed by OpenAI, is a state-of-the-art transformer-based model designed to generate human-like text. It can be used for various natural language processing (NLP) tasks such as text completion, summarization, and creative writing.

## Overview

The example in this repository demonstrates how to leverage the pre-trained `gpt2-medium` model from the Hugging Face `transformers` library to generate text based on a given prompt. By following this guide, you will learn how to:

1. Load a pre-trained GPT-2 model and tokenizer.
2. Encode an input text prompt.
3. Generate text using the model.
4. Decode and display the generated text.

This guide is intended for users who have a basic understanding of Python and are interested in NLP and machine learning.

## Approach

### Step 1: Import Libraries and Load Model

To begin, we import the necessary libraries and load the pre-trained GPT-2 model and tokenizer from the Hugging Face `transformers` library. The `gpt2-medium` model is chosen for its balance between performance and computational requirements.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```

### Step 2: Encode Input Text

Next, we prepare the input text for the model by encoding it into input IDs. This step converts the human-readable text into a format that the model can process.

```python
# Encode input text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
```

### Step 3: Generate Text

With the input text encoded, we use the model to generate text. We specify the maximum length of the generated text and the number of sequences to return. This allows us to control the length and variability of the output.

```python
# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
```

### Step 4: Decode and Print the Generated Text

Finally, we decode the output IDs back into human-readable text and print it. This step transforms the model's output into a format that is understandable and usable.

```python
# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### Example Output

Running the above code might produce an output like this:

```
Once upon a time, there was a little girl who lived in a village near the forest. Whenever she went out, the little girl wore a red riding cloak, so everyone in the village called her Little Red Riding Hood.
```

## Conclusion

This example demonstrates a straightforward approach to using GPT-2 for text generation. By experimenting with different input texts and parameters, you can explore the capabilities of GPT-2 and adapt it to various applications in natural language processing.

# Thank You!
