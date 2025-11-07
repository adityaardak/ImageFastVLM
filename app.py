import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- CPU-only config ----
MID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # special image token id used by FastVLM

tok = None
model = None

def load_model():
    global tok, model
    if tok is None or model is None:
        print("Loading model (CPU)…")
        tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
        # Force CPU + float32 (fp16 is unsafe on CPU)
        model = AutoModelForCausalLM.from_pretrained(
            MID,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("Model loaded successfully on CPU!")
    return tok, model

def caption_image(image, custom_prompt=None):
    """
    Generate a caption for the input image (CPU-only).
    """
    if image is None:
        return "Please upload an image first."

    try:
        tok, model = load_model()

        # Convert image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = custom_prompt if custom_prompt else "Describe this image in detail."

        # Single-turn chat with an <image> placeholder
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Split around the literal "<image>"
        pre, post = rendered.split("<image>", 1)

        # Tokenize text around the image token
        pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids

        # Derive device/dtype from the loaded model (CPU here, but future-proof)
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        # Insert IMAGE token id at placeholder position
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype, device=model_device)
        input_ids = torch.cat(
            [pre_ids.to(model_device), img_tok, post_ids.to(model_device)],
            dim=1
        )
        attention_mask = torch.ones_like(input_ids, device=model_device)

        # Preprocess image using model's vision tower
        px = model.get_vision_tower().image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device=model_device, dtype=model_dtype)

        # Generate caption (deterministic)
        with torch.no_grad():
            out = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=128,
                do_sample=False,  # temperature is ignored when sampling is off
            )

        # Decode and slice to the assistant part if present
        generated_text = tok.decode(out[0], skip_special_tokens=True)
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:", 1)[-1].strip()
        elif "assistant" in generated_text:
            response = generated_text.split("assistant", 1)[-1].strip()
        else:
            response = generated_text.strip()

        return response

    except Exception as e:
        return f"Error generating caption: {str(e)}"

# ---- Gradio UI (CPU) ----
with gr.Blocks(title="FastVLM Image Captioning (CPU)") as demo:
    gr.Markdown(
        """
        # 🖼️ FastVLM Image Captioning (CPU)
        Upload an image to generate a detailed caption using Apple's FastVLM-0.5B.
        This build runs on **CPU only**. Expect slower generation than GPU.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image", elem_id="image-upload")
            custom_prompt = gr.Textbox(
                label="Custom Prompt (Optional)",
                placeholder="Leave empty for default: 'Describe this image in detail.'",
                lines=2
            )
            with gr.Row():
                clear_btn = gr.ClearButton([image_input, custom_prompt])
                generate_btn = gr.Button("Generate Caption", variant="primary")

        with gr.Column():
            output = gr.Textbox(
                label="Generated Caption",
                lines=8,
                max_lines=15,
                show_copy_button=True
            )

    generate_btn.click(fn=caption_image, inputs=[image_input, custom_prompt], outputs=output)

    # Also generate on image upload if no custom prompt
    def _auto_caption(img, prompt):
        return caption_image(img, prompt) if (img is not None and not prompt) else None

    image_input.change(fn=_auto_caption, inputs=[image_input, custom_prompt], outputs=output)

    gr.Markdown(
        """
        ---
        **Model:** [apple/FastVLM-0.5B](https://huggingface.co/apple/FastVLM-0.5B)  
        **Note:** CPU-only run. For speed, switch to a CUDA GPU build or a GPU Space.
        """
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )