import os
import requests
import tiktoken
import numpy as np
from pathlib import Path

filepath = "/Users/juliencarbonnell/Desktop/nearData/nearBlog"
paths = [str(x) for x in Path(filepath).glob("**/*")]
data = ""

# Convert file to UTF-8 encoding
def convert_to_utf8(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
        try:
            decoded_content = content.decode('utf-8')
        except UnicodeDecodeError:
            decoded_content = content.decode('latin1')  # Try another encoding if UTF-8 fails

    with open(filepath, 'wb') as f:
        f.write(decoded_content.encode('utf-8'))

# loop over files to encode them as utf-8
for root, dirs, files in os.walk(filepath):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        convert_to_utf8(file_path)
        print(f"Converted {file_path} to UTF-8 encoding.")

# load all the text files in a single string
for i in range(len(paths)):
    print("reading " + os.path.basename(paths[i]))
    with open(paths[i], "r") as f:
        data = data + "\n" + f.read()

n = len(data)
# train val split
train_data = data[:int(n*0.9)] #90%
val_data = data[int(n*0.9):] #10%

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
