from safetensors.torch import load_file, safe_open, save
import torch, os

init = ['encoder.gate_dense.bias', 'encoder.gate_dense.weight', 'encoder.image_dense.bias', 'encoder.image_dense.weight', 'encoder.mha_layer.in_proj_bias', 'encoder.mha_layer.in_proj_weight', 'encoder.mha_layer.out_proj.bias', 'encoder.mha_layer.out_proj.weight']


def generate_init_weights(model_path, to):
    model = torch.load(model_path)
    update = {k: v.cpu() for k, v in model.items() if k in init}
    torch.save(update, to)


def merge_weights():
    f_a = r"C:\Users\yushi\.cache\huggingface\hub\models--declare-lab--flan-alpaca-base\snapshots\c7a05e47b9704254fad4832b9a1efc4123799c65\model.safetensors"
    f_b = "models/mha.bin"
    save_to = r"C:\Users\yushi\.cache\huggingface\hub\models--declare-lab--flan-alpaca-base\snapshots\c7a05e47b9704254fad4832b9a1efc4123799c65\model.safetensors2"
    with safe_open(f_a, framework="pt") as f:
        metadata = f.metadata()
        print(metadata)
    a = load_file(f_a)
    b = torch.load(f_b)
    a.update(b)
    open(save_to, 'wb').write(save(a, metadata))
    os.renames(f_a, f_a + "_bak")
    os.renames(save_to, f_a)


if __name__ == '__main__':
    # generate_init_weights("models/mm-cot-base-rationale/pytorch_model.bin", "init.bin")
    # print(torch.load("init.bin").keys())
    f_a = r"/root/.cache/huggingface/hub/models--declare-lab--flan-alpaca-base/snapshots/c7a05e47b9704254fad4832b9a1efc4123799c65/model.safetensors"
    f_b = "/root/autodl-fs/init.bin"
    save_to = r"/root/.cache/huggingface/hub/models--declare-lab--flan-alpaca-base/snapshots/c7a05e47b9704254fad4832b9a1efc4123799c65/model.safetensors2"
    with safe_open(f_a, framework="pt") as f:
        metadata = f.metadata()
        print(metadata)
    a = load_file(f_a)
    b = torch.load(f_b)
    a.update(b)
    open(save_to, 'wb').write(save(a, metadata))
    os.renames(f_a, f_a + "_no_init")
    os.renames(save_to, f_a)