# fix_encoding.py
with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed encoding in main.py")

"""
Fix encoding issues in all Python files
"""
import os
import sys

files_to_fix = [
    'main.py',
    'src/models.py',
    'src/data_utils.py', 
    'src/training.py',
    'src/explainability.py'
]

print("Fixing encoding in Python files...")
print()

for filepath in files_to_fix:
    if not os.path.exists(filepath):
        print(f"Skip {filepath} (not found)")
        continue
        
    try:
        # Read with error handling
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Write back as clean UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"OK {filepath}")
    except Exception as e:
        print(f"ERROR {filepath}: {e}")

print()
print("Done! Now try running:")
print("python simple_test.py")