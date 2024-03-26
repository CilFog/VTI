import sys
import runpy
import os

if __name__ == "__main__":
    # Convert file path to module path
    file_path = sys.argv[1]
    
    # Find the position of "VTI"
    VTI = 'VTI/'
    vti_pos = file_path.find(VTI)
    
    module_name = file_path[vti_pos+len(VTI):].replace('/', '.').replace('\\', '.').rstrip('.py')
    # Assuming the script is run from the project root
    runpy.run_module(module_name, run_name="__main__")