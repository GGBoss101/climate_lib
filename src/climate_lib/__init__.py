import importlib
import os
import sys
import pkgutil

# Find the path of this folder
_package_dir = os.path.dirname(__file__)

# Loop through and import every Python file as a module object
for _, _module_name, _ in pkgutil.iter_modules([_package_dir]):
    
    # 1. Dynamically import the file relative to this folder
    _module = importlib.import_module(f".{_module_name}", package=__name__)
    
    # 2. Safely attach the module object to this package namespace
    setattr(sys.modules[__name__], _module_name, _module)