import os
import sys

path = r'C:\Users\10225\Downloads\KB\extracted'
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for f in files[:5]:
        print(f'{subindent}{f}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')