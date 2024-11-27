import torch
import torch.nn as nn
from numpy import dtype
from transformers import CLIPVisionModel, SiglipVisionModel, CLIPVisionConfig, SiglipVisionConfig, AutoProcessor
from transformers import LlamaForCausalLM, SuppressTokensLogitsProcessor
from .muse.muse import VQGANModel
import numpy as np
def read_image_to_tensor_v2(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return img .astype(np.float32)

class VisualTokenizer():
    def __init__(self, device_map, ckpts = "/home/niudt/vqlm/muse/ckpts/laion"):
        net = VQGANModel.from_pretrained(ckpts).to(device_map)
        net = net.eval()
        self.tokenizer = net

    def tokenize_images(self, prompt):
        # # here tokenize the image in the list
        # selected_images = image_list
        # try:
        #     prompt_images = [read_image_to_tensor_v2(ids) for ids in selected_images]
        # except:
        #     return None
        # prompt = np.stack(prompt_images, axis=0)
        with torch.no_grad():
            _, prompt_tokens = self.tokenizer.encode(prompt)
            # prompt_tokens = torch.reshape(prompt_tokens, [1, -1])
        return prompt_tokens

    def detokenize(self, tokens, save_path, idx):
        s = 1
        # tokens = torch.reshape(tokens, [16, -1])
        plt.figure(figsize=(12, 12))
        with torch.no_grad():
            re_constructed = self.tokenizer.decode_code(tokens)
        for i in range(re_constructed.shape[0]):
            recon_img = torch.clamp(re_constructed[i],
                                    0.0, 1.0
                                    )
            plt.subplot(1, 1, i + 1)
            plt.imshow((((recon_img).permute(1, 2, 0).detach().cpu().numpy() * 255)).astype(np.int32))
            plt.grid(False)
            plt.axis('off')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig('{}/{}_{}.png'.format(save_path, 'temp', str(idx)))

class LVMVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            if 'siglip' in self.vision_tower_name:
                self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = AutoProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = LlamaForCausalLM.from_pretrained(
            '/scratch/partial_datasets/lvma/LRM/cvpr/ckpts-lvm/7b', device_map=device_map)



        self.vision_tower.requires_grad_(False)
        self.vqgan = VisualTokenizer(device_map)
        self.linear_proj = nn.Linear(4096, 1024, bias=True)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, split_sizes):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # here vqgan
            batch_size = len(split_sizes)
            vq_sequence = self.vqgan.tokenize_images(images.to(device=self.device, dtype = self.vqgan.tokenizer.dtype))
            vq_sequence = vq_sequence.view(batch_size, -1, 256)
            vq_sequence = vq_sequence.view(batch_size, -1)
            image_forward_outs = self.vision_tower(vq_sequence.to(device=self.device), output_hidden_states=True)['hidden_states'][-1]
            image_forward_outs = self.linear_proj(image_forward_outs)
            hidden_size = image_forward_outs.shape[-1]
            image_forward_outs = image_forward_outs.view(batch_size, -1, 256, hidden_size)
            image_features = image_forward_outs.view(-1, 256, hidden_size)


        return image_features # 32 * 576 * 1024 // 32 * 256 * 4096

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
