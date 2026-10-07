import torch #ejecutar Operaciones matematicas del modelo de IA
from diffusers import AutoPipelineForText2Image
# Diffusers es una libreria espacializada en modelos generativos

print("Cargando el modelo ... ")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32 # float16 o float 32
    #torch.floatXX, indica la precisión numerica con que se cargan los pesos (decimales, la precisión para generar la imagen)
    
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")
negative_prompt= "ugly, deformed, noisy, blurry, distorted, low quality, bad anatomy, bad proportions, duplicate, watermark, signature, text, cropped, extra limbs, missing limbs"

print("Generando imagen")

imagen = modelo(
    prompt= prompt, #Instrucciones para crear la img
    negative_prompt= negative_prompt,
    num_inference_steps=10, # Las veces que va a tratar de crear o refinar la imagen
    guidance_scale=2.0, # Que tan fiel debe ser al prompt (Creatividad, temperatura)
    heigth=1024,
    width=1024
    
).images[0]

imagen.save("imagen.png")

print("imagen guardada.")



