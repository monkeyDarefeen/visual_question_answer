#LLAVA QA
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

def load_llava_model(model_path):
    # Setup device for GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load processor and model
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(device)
    return model,processor