import io
import gc
import torch
from flask import Flask, request, send_file
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
from diffusers.utils import logging
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

# Configure logging
logging.set_verbosity_error()

def generate_image_from_text(text):
    model_path = "./models/stable-diffusion-2-1"
    
    try:
        # Check if FP16 is supported
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
            
        if not torch.cuda.is_bf16_supported():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16

        # Load pipeline with memory optimizations
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            variant="fp16",
            safety_checker=None,  # Disable if not needed
            requires_safety_checker=False,
            local_files_only=True
        )
        
        # Enable memory optimizations
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing(1)  # Aggressive slicing
        pipe.enable_sequential_cpu_offload()  # Offload to CPU
        pipe.enable_model_cpu_offload()  # Additional offloading
        
        # Generate with reduced memory footprint
        generator = torch.Generator(device="cuda").manual_seed(42)
        image = pipe(
            prompt=text,
            num_inference_steps=20,  # Reduced from default 50
            guidance_scale=7.5,
            height=384,  # Reduced from 512
            width=384,
            generator=generator
        ).images[0]

        return image

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            return f"Error: {str(e)}. Try a shorter prompt or reduce complexity."
        raise

    finally:
        # Cleanup
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()
        gc.collect()

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