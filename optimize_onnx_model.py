import os
import onnx
from onnx import optimizer
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", help="Path to directory to .onnx models.", default=None, type=str)
    return parser
    
if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    
    filenames = [filename for filename in os.listdir(args.output_dir) if filename.endswith(".onnx") and not filename.startswith("opt-")]
    
    for filename in filenames:
    
        source = os.path.join(args.output_dir, filename)
        target = os.path.join(args.output_dir, "opt-{}".format(filename))
        
        onnx_model = onnx.load(source)
        passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
        optimized_model = optimizer.optimize(onnx_model, passes)
        onnx.save(optimized_model, target)
        