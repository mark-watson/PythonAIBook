import ollama

# Specify the path to the image file to be analyzed
image_path = 'ticket.png'

# Send the image to the vision-capable model for a detailed description
response = ollama.chat(
    model='qwen3.5:0.8b', # Ensure you use a vision-capable model
    messages=[{
        'role': 'user',
        'content': 'Describe this image in detail',
        'images': [image_path]
    }],
    think=False # Suppresses the <think> reasoning block
)

# Print the model's descriptive analysis of the image
print(response.message.content)
