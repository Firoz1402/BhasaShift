import io
import gc
import torch
from flask import Flask, request, send_file
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

def translate_assamese_to_english(assamese_text):
    # Load translation components
    model_path = "./models/Bhasashift-v1/final-model"
    tokenizer = AutoTokenizer.from_pretrained("./models/Bhasashift-v1/final-model-tokeniser")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

    # Process text
    inputs = tokenizer(
        assamese_text,
        max_length=256,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate translation
    outputs = model.generate(**inputs)
    english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return english_text

def generate_image_from_text(text):
    # Load pipeline with memory optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    
    # Generate image
    image = pipe(text, num_inference_steps=25, guidance_scale=7.5).images[0]

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return image

@app.route('/generate/image/assamese', methods=['POST'])
def handle_request():
    # Get input text
    data = request.get_json()
    if not data or 'text' not in data:
        return "Missing text parameter", 400
    
    try:
        # Step 1: Translate text
        translated_text = translate_assamese_to_english(data['text'])
        
        # Step 2: Generate image
        generated_image = generate_image_from_text(translated_text)
        
        # Convert image to bytes
        img_io = io.BytesIO()
        generated_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            return "Insufficient GPU memory - try a shorter prompt", 500
        return f"Error: {str(e)}", 500
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=False)