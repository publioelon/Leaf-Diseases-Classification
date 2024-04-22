import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.system("source /opt/intel/openvino/bin/setupvars.sh")  # For Linux
os.system("C:\\Program Files (x86)\\Intel\\openvino_2022\\bin\\setupvars.bat")  # For Windows

# Define the setupvars.bat path according to where you installed it in your system


import subprocess

# Define paths
saved_model_path = 'path/to/savedmodel'  
output_dir = 'output/path/to/IRfile' # IR output path this will output a .xml containing the model and a .bin file containing the model weights            
input_shape = "[1,229,229,3]" # Always use [] to define the model input shape, MobileNets will use 224, 224. So it needs to be [1, 224, 224, 3]                        

# Form the Model Optimizer command
mo_command = f"mo --saved_model_dir {saved_model_path} --input_shape {input_shape} --output_dir {output_dir}"

# Run the command
result = subprocess.run(mo_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output (optional)
print(result.stdout)
print(result.stderr)
