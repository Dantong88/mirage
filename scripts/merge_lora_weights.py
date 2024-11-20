import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False, default='/home/niudt/project/llarva_more/mirage/checkpoints/mirage_qformer_ft_lora')
    parser.add_argument("--model-base", type=str, required=False, default='/home/niudt/project/llarva_more/mirage/ckpts/mirage-llama3.1-8.3B_main')
    parser.add_argument("--save-model-path", type=str, required=False, default='/home/niudt/project/llarva_more/mirage/checkpoints/mirage_qformer_ft_')

    args = parser.parse_args()

    merge_lora(args)