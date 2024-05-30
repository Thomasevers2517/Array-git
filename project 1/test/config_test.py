import sys
import os
path_to_add = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path_to_add)
# Add the src directory to the sys.path
sys.path.insert(0, os.path.abspath(path_to_add))
