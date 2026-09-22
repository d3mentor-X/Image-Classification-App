from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()

image = Image.open("test2.jpg").convert("RGB")

transform = weights.transforms()
input_image = transform(image).unsqueeze(0)

with __import__("torch").no_grad():
    output = model(input_image)

probabilities = output.softmax(1)[0]
class_id = probabilities.argmax().item()

print("Prediction:", weights.meta["categories"][class_id])
print("Confidence:", f"{probabilities[class_id].item() * 100:.2f}%")