import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import os
import torch

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model_path = args.save_model_path

    if args.save_merge:
        model.save_pretrained(args.save_model_path)
        tokenizer.save_pretrained(args.save_model_path)

    if args.save_split:
        save_dir = os.path.join(os.path.dirname(model_path), 'split', 'llarva-part')
        os.makedirs(save_dir, exist_ok=True)

        qformer = model.model.qformer
        # retriever = model.model.retriever
        mm_projector = model.model.mm_projector
        torch.save(qformer.state_dict(), os.path.join(save_dir, "qformer.pth"))
        # torch.save(retriever, os.path.join(save_dir, "retriever.pth"))
        torch.save(mm_projector.state_dict(), os.path.join(save_dir, "mm_projector.pth"))

        del model.model.qformer
        del model.model.vision_tower
        del model.model.mm_projector

        save_dir = os.path.join(os.path.dirname(model_path), 'split', 'llarva-main')
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False, default='/scratch/partial_datasets/niudt/project/llarva_v2/ckpts/lora/mirage-lora-lvm-4096-close_jar_initial_test_ws8_b32_10ep_5e-5/checkpoint-1000-llarva-lora')
    parser.add_argument("--model-base", type=str, required=False, default='/home/niudt/project/llarva_more/mirage/ckpts/mirage-llama3.1-8.3B_main')
    parser.add_argument("--save-model-path", type=str, required=False, default='/scratch/partial_datasets/niudt/project/llarva_v2/ckpts/merged/mirage-lvm-4096-close_jar_initial_test_ws8_b32_10ep_5e-5-checkpoint-1000')
    parser.add_argument("--save-merge", type=bool, required=False,
                        default=True)
    parser.add_argument("--save-split", type=bool, required=False,
                        default=False)

    args = parser.parse_args()

    merge_lora(args)