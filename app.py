import io
import gc
import torch
from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, VisionEncoderDecoderModel, ViTImageProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
from diffusers.utils import logging
app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)

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

@app.route('/translate/english-to-assamese-old', methods=['POST'])
def translate_english_to_assamese():
    # Get input text
    data = request.get_json()
    if not data or 'text' not in data:
        return "Missing text parameter", 400

    english_text = data['text']

    try:
        # Load translation components
        model_path = "./models/Bhasashift-v1/final-model"
        tokenizer = AutoTokenizer.from_pretrained("./models/Bhasashift-v1/final-model-tokeniser")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

        # Process text
        inputs = tokenizer(
            english_text,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        # Generate translation
        outputs = model.generate(**inputs)
        assamese_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cleanup
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return {"translated_text": assamese_text}, 200

    except Exception as e:
        return {"error": str(e)}, 500

# Endpoint 1: Translate English to Assamese
@app.route('/translate/english-to-assamese', methods=['POST'])
def translate_english_to_assamese_v2():
    data = request.get_json()
    if not data or 'text' not in data:
        return "Missing text parameter", 400

    english_text = data['text']

    try:
        model_path = "./models/Bhasashift-v2/english-to-assamese-translator/model"
        tokenizer_path = "./models/Bhasashift-v2/english-to-assamese-translator/tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

        inputs = tokenizer(
            english_text,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(**inputs)
        assamese_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return {"translated_text": assamese_text}, 200

    except Exception as e:
        return {"error": str(e)}, 500

# Endpoint 2: Translate Assamese to English
@app.route('/translate/assamese-to-english', methods=['POST'])
def translate_assamese_to_english_v2():
    data = request.get_json()
    if not data or 'text' not in data:
        return "Missing text parameter", 400

    assamese_text = data['text']

    try:
        model_path = "./models/Bhasashift-v2/assamese-to-english-translator/model"
        tokenizer_path = "./models/Bhasashift-v2/assamese-to-english-translator/tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

        inputs = tokenizer(
            assamese_text,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(**inputs)
        english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return {"translated_text": english_text}, 200

    except Exception as e:
        return {"error": str(e)}, 500

# Endpoint 3: Generate Image in Assamese
@app.route('/generate/image/assamese', methods=['POST'])
def generate_image_assamese():
    data = request.get_json()
    if not data or 'text' not in data:
        return "Missing text parameter", 400

    assamese_text = data['text']

    try:
        # Step 1: Translate Assamese to English
        model_path = "./models/Bhasashift-v2/assamese-to-english-translator/model"
        tokenizer_path = "./models/Bhasashift-v2/assamese-to-english-translator/tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

        inputs = tokenizer(
            assamese_text,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(**inputs)
        english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        # Step 2: Generate Image using Stable Diffusion
        model_path = "./models/stable-diffusion-2-1"

        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        ).to("cuda")

        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()
        pipe.enable_model_cpu_offload()

        generator = torch.Generator(device="cuda").manual_seed(42)
        image = pipe(
            prompt=english_text,
            num_inference_steps=20,
            guidance_scale=7.5,
            height=384,
            width=384,
            generator=generator
        ).images[0]

        # Convert image to bytes
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        # Encode the image as a blob
        image_blob = img_io.getvalue()

        del pipe, image
        torch.cuda.empty_cache()
        gc.collect()

        return {"image_blob": image_blob.hex()}, 200

    except Exception as e:
        return {"error": str(e)}, 500

# Endpoint 4: Image Caption in Assamese
@app.route('/caption/image/assamese', methods=['POST'])
def caption_image_assamese():
    if 'file' not in request.files:
        return "Missing image file", 400

    image_file = request.files['file']
    image = Image.open(image_file.stream)

    try:
        # Step 1: Generate English Caption using BLIP model
        caption_model_path = "./models/blip-image-captioning"

        # Load the BLIP model and processor
        caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_path).to("cuda")
        caption_processor = ViTImageProcessor.from_pretrained(caption_model_path)

        # Load the tokenizer explicitly
        caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_path)

        # Preprocess the image
        pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values.to("cuda")

        # Generate caption using BLIP
        output_ids = caption_model.generate(pixel_values)
        english_caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("English Caption:")
        print(english_caption)

        # Step 2: Translate English Caption to Assamese
        model_path = "./models/Bhasashift-v2/english-to-assamese-translator/model"
        tokenizer_path = "./models/Bhasashift-v2/english-to-assamese-translator/tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

        inputs = tokenizer(
            english_caption,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(**inputs)
        assamese_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return {"caption": assamese_caption}, 200

    except Exception as e:
        return {"error": str(e)}, 500

# Endpoint 5: Gemini API with Assamese Prompts
@app.route('/gemini/assamese', methods=['POST'])
def gemini_assamese():
    data = request.get_json()
    if not data or 'text' not in data:
        return "Missing text parameter", 400

    assamese_prompt = data['text']

    try:
        # Step 1: Translate Assamese to English
        model_path = "./models/Bhasashift-v2/assamese-to-english-translator/model"
        tokenizer_path = "./models/Bhasashift-v2/assamese-to-english-translator/tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

        inputs = tokenizer(
            assamese_prompt,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(**inputs)
        english_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        # Step 2: Call Gemini 2.0 Flash API
        import requests

        gemini_api_url = "https://gemini-api.example.com/v1/generate"  # Replace with the actual API endpoint
        gemini_api_key = "your_gemini_api_key"  # Replace with your actual API key

        headers = {
            "Authorization": f"Bearer {gemini_api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(gemini_api_url, json={"prompt": english_prompt}, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            return {"error": response_data.get("error", "Unknown error")}, response.status_code

        gemini_response = response_data.get("response", "")

        # Step 3: Translate Gemini Response to Assamese
        model_path = "./models/Bhasashift-v2/english-to-assamese-translator/model"
        tokenizer_path = "./models/Bhasashift-v2/english-to-assamese-translator/tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

        inputs = tokenizer(
            gemini_response,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(**inputs)
        assamese_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return {"response": assamese_response}, 200

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)