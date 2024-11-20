import json

from PIL import Image
import argparse
import torch
import os
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.utils import disable_torch_init

class DEMO:
    def __init__(self, model_path):
        model_name = get_model_name_from_path(model_path)
        disable_torch_init()
        model_name = os.path.expanduser(model_name)
        self.tokenizer, self.model, self.image_processor, _ = \
            load_pretrained_model(model_path=model_path, model_base=None, model_name=model_name, device="cuda")
        self.model.eval_mode = True



    @torch.inference_mode()
    def demo(self, image_paths, prompt, num_retrievals=1):
        self.conv = conv_templates["llama3"].copy()
        clip_images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_tensor = image_tensor.to(dtype=torch.float16)
            clip_images.append(image_tensor)

        qformer_text_input = self.tokenizer(prompt, return_tensors='pt')["input_ids"].to(self.model.device)

        N = len(clip_images)
        img_str = DEFAULT_IMAGE_TOKEN * N + "\n"
        inp = img_str + prompt

        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        N = len(clip_images)
        self.tokenizer.pad_token_id = 128002
        if N <= 200:
            batch_clip_imaegs = [torch.stack(clip_images).to(self.model.device)]
            with torch.inference_mode():
                output_ret, output_ids = self.model.generate(
                    input_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                    clip_images=batch_clip_imaegs,
                    qformer_text_input=qformer_text_input,
                    relevance=None,
                    num_retrieval=num_retrievals,
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True)
        else:
            # batch size is too large, split into smaller batches
            batch_clip_imaegs = [torch.stack(clip_images)]
            with torch.inference_mode():
                output_ret, output_ids = self.model.batch_generate(
                    input_ids,
                    clip_images=batch_clip_imaegs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    qformer_text_input=qformer_text_input,
                    relevance=None,
                    num_retrieval=num_retrievals,
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True)

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/niudt/project/llarva_more/mirage/checkpoints/mirage_qformer_ft")
    parser.add_argument("--max-num-retrievals", type=int, default=32)
    parser.add_argument('--test-anns', type=str, default='/home/yuvan/project/vhs_exploration_nov16/close_jar/val_4148.json')
    args = parser.parse_args()

    test_anns = args.test_anns
    test_anns = json.load(open(test_anns))

    model = DEMO(args.model_path)
    for ann in tqdm(test_anns):
        image_paths = ann['image']
        prompt = ann['conversations'][0]['value'][8:]
        text_output = model.demo(image_paths, prompt, args.max_num_retrievals)
        print('---Input---')
        print("Prompt:", prompt)
        print("Images:", image_paths)
        print('---Output---')
        print("Text Output:", text_output)
        print('---GT---')
        print("Text Output:", ann['conversations'][1]['value'])
