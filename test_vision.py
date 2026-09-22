from transformers import pipeline

print("Loading AI model...")

pipe = pipeline(
    "image-text-to-text",
    model="HuggingFaceTB/SmolVLM-256M-Instruct"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "test.jpg"
            },
            {
                "type": "text",
                "text": "Analyze this image in detail. Identify the main object or objects, describe their appearance, explain the scene, and mention important visible details."
            }
        ]
    }
]

print("Analyzing image...")

result = pipe(
    text=messages,
    max_new_tokens=200,
    return_full_text=False
)

print("\n===== IMAGE ANALYSIS =====\n")
print(result[0]["generated_text"])