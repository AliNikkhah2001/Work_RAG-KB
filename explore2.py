import os
import sys
import json

path = r'C:\Users\10225\Downloads\KB\extracted'
structure = {}

for root, dirs, files in os.walk(path):
    rel_path = os.path.relpath(root, path)
    if rel_path == '.':
        rel_path = 'root'
    structure[rel_path] = {
        'dirs': dirs,
        'files': files
    }

with open(r'C:\Users\10225\Downloads\KB\structure.json', 'w', encoding='utf-8') as f:
    json.dump(structure, f, ensure_ascii=False, indent=2)

print("Done")