# Referenced from: https://github.com/gortizji/tangent_task_arithmetic/blob/main/src/modeling.py

import open_clip
import torch
from torchvision import transforms

from src import utils
from timm.data.transforms_factory import transforms_imagenet_train

class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f"Loading {args.model} pre-trained weights.")
        if "__pretrained__" in args.model:
            name, pretrained = args.model.split("__pretrained__")
        elif "__init__" in args.model:
            print("Using random initialization.")
            name, pretrained = args.model.split("__init__")[0], None
        else:
            name = args.model
            pretrained = "openai"
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )

        # get attention list
        module_name_list = []
        self.get_all_layer_name_by_class_name_list(self.model, [], module_name_list, 'MultiheadAttention')
        module_name_list = list(filter(lambda x: 'visual.transformer' in x, module_name_list))

        # monkey patch attention
        for module_name in module_name_list:
            attn = get_module_by_name(self.model, module_name)
            new_attn = CustomMultiheadAttention(attn)
            for p in new_attn.parameters():
                p = p.requires_grad_(False)
            set_module_by_name(self.model, module_name, new_attn)

        if args.auto_aug is not None:
            self.train_preprocess = transforms_imagenet_train(
                img_size=224,
                auto_augment=args.auto_aug,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )

            if args.train_dataset == "MNISTVal":
                self.train_preprocess =  transforms.Compose([
                    transforms.Resize(224),
                    transforms.Grayscale(num_output_channels=3),
                    transforms_imagenet_train(
                        img_size=224,
                        auto_augment=args.auto_aug,
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)
                    )
                ])

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu", weights_only=False)
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )
        self.model.load_from_state_dict(state_dict)

    def get_all_layer_name_by_class_name_list(self, module, current_module_name, module_name_list, class_name):
        if module.__class__.__name__ == class_name:  # residual block
            module_name_list.append('.'.join(current_module_name))
        else:
            for k in module._modules.keys():
                self.get_all_layer_name_by_class_name_list(getattr(module, k), current_module_name + [k],
                                                           module_name_list, class_name)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)

from functools import reduce
def get_module_by_name(module, access_string):
    # https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

def set_module_by_name(module, access_string, new_module):
    names = access_string.split(sep='.')
    for name in names[:-1]:
        module = getattr(module, name)
    setattr(module, names[-1], new_module)


def custom_scaled_dot_product_attention(q, k, v, attn_mask=None):
    d_k = q.size(-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if attn_mask is not None:
        attn_scores += attn_mask

    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    return attn_output, attn_weights

import torch.nn as nn

class CustomMultiheadAttention(nn.Module):
    '''
    convert nn.MultiheadAttention to CustomMultiheadAttention

    MultiheadAttention
        Q, K, V are concated

    CustomMultiheadAttention
        Q, K, V are splitted

    Note
        True: torch.allclose(real_output[0][0], custom_output[0][0], atol=1e-08)
        False: torch.allclose(real_output[0][0], custom_output[0][0], atol=1e-09)
    '''

    def __init__(self, multihead_attn):
        super().__init__()
        Q_weight, K_weight, V_weight = (
            multihead_attn.in_proj_weight
            .detach().requires_grad_(False)
            .chunk(3, dim=0)
        )
        Q_bias, K_bias, V_bias = (
            multihead_attn.in_proj_bias
            .detach().requires_grad_(False)
            .chunk(3, dim=0)
        )
        O_bias = (
            multihead_attn.out_proj.bias
            .detach().requires_grad_(False)
        )
        O_weight = (
            multihead_attn.out_proj.weight
            .detach().requires_grad_(False)
        )

        fan_in, fan_out = Q_weight.shape

        self.Q = nn.Linear(fan_in, fan_out)
        self.K = nn.Linear(fan_in, fan_out)
        self.V = nn.Linear(fan_in, fan_out)
        self.O = nn.Linear(fan_in, fan_out)

        self.head_dim = multihead_attn.head_dim
        self.num_heads = multihead_attn.num_heads

        with torch.no_grad():
            self.Q.weight.copy_(Q_weight)
            self.K.weight.copy_(K_weight)
            self.V.weight.copy_(V_weight)
            self.O.weight.copy_(O_weight)

            self.Q.bias.copy_(Q_bias)
            self.K.bias.copy_(K_bias)
            self.V.bias.copy_(V_bias)
            self.O.bias.copy_(O_bias)

        del Q_weight, K_weight, V_weight, O_weight, Q_bias, K_bias, V_bias, O_bias

    def forward(self, q, k, v, need_weights=False, attn_mask=None):
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        tgt_len, bsz, embed_dim = q.shape
        src_len, *_ = k.shape

        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output, attn_weights = custom_scaled_dot_product_attention(q, k, v, attn_mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn_output = self.O(attn_output)
        return attn_output.transpose(0, 1), attn_weights if need_weights else None
