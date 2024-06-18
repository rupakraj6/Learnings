from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_response(input_text):
    # Load the model
    model_name = "your_model_name_here"  # replace with your model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Encode the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response
    outputs = model.generate(inputs, max_length=150, num_return_sequences=5, temperature=1.0)

    # Decode the response
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return responses

# Test the function
input_text = "your_input_text_here"  # replace with your input text
responses = generate_response(input_text)
print(responses)


