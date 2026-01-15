import torch
import onnx, onnxruntime
import yaml
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
import shutil

from .model import PartSegmentationModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--input-size", type=int,
                        default=320)
    parser.add_argument("--save-path",  type=Path, default="output")

    return parser.parse_args()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    args = parse_args()
    args.save_path.mkdir(parents=True,
                         exist_ok=True)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.save_path / "config.yaml", 'w') as f:
        yaml.safe_dump(config, f)

    shutil.copyfile(config['data']['labels_change_class'],
                    args.save_path / 'labels_change_class.json')

    shutil.copyfile(config['data']['labels_to_idx'],
                    args.save_path / 'labels_to_idx.json')

    model = PartSegmentationModel.load_from_checkpoint(args.ckpt_path, **config["model"],
                                                       n_classes=config["data"]["n_classes"],
                                                       map_location='cpu').eval()

    torch.save(model.model.state_dict(),
               args.save_path / "parts_model_state_dict.pt")

    print("-----------------------")
    try:
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
        exported_program = torch.export.export(model.model,
                                               (dummy_input, )
                                               )
        torch.export.save(exported_program,
                          args.save_path / 'parts_model_exported.pt2')
        print("Exported program")
    except Exception as e:
        print(f"Got error exporting model as a torch program. {e}")

    print("-----------------------")
    try:
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
        scripted_module = torch.jit.script(model.model,
                                           example_inputs=dummy_input)
        torch.jit.save(scripted_module,
                       args.save_path / 'parts_model_scripted.pt')
        traced_model = torch.jit.load(args.save_path / 'parts_model_scripted.pt')
        if torch.allclose(model.model(dummy_input), traced_model(dummy_input)):
            print("Exported jit script")
        else:
            print("Different output from the scripted model")
    except Exception as e:
        print(f"Got error script the model to jit {e}")

    print("-----------------------")

    try:
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
        scripted_module = torch.jit.trace(model.model,
                                          example_inputs=dummy_input)
        torch.jit.save(scripted_module,
                       args.save_path / 'parts_model_traced.pt')

        traced_model = torch.jit.load(args.save_path / 'parts_model_traced.pt')
        if torch.allclose(model.model(dummy_input), traced_model(dummy_input)):
            print("Exported jit trace")
        else:
            print("Different output from the traced model")
    except Exception as e:
        print(f"Got error tracing the model to jit {e}")

    print("-----------------------")
    try:
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
        torch_out = model(dummy_input)
        torch.onnx.export(model,
                          dummy_input,
                          args.save_path / "parts_model.onnx",
                          verbose=False,
                          input_names=["input"],
                          output_names=["output"],
                          export_params=True,
                          )

        onnx_model = onnx.load(args.save_path / "parts_model.onnx")
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession(args.save_path / "parts_model.onnx", providers=["CPUExecutionProvider"])
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    except Exception as e:
        print(f"Got error exporting to the onnx. {e}")
