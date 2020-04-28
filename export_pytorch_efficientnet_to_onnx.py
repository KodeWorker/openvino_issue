import os
import torch 
from efficientnet_pytorch import EfficientNet
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", help="Path to directory to .onnx models.", default=None, type=str)
    return parser

if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    
    for model_name in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']:
        onnx_file = os.path.join(args.output_dir, "{}.onnx".format(model_name)) 
        
        model = EfficientNet.from_pretrained(model_name)    
        model.set_swish(memory_efficient=False) #!!!
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        input_names = [ "input" ]
        output_names = [ "output" ]
        
        torch.onnx.export(model, dummy_input, onnx_file, verbose=True, input_names=input_names, output_names=output_names,
        keep_initializers_as_inputs=True)
    