# Hashing a file using SHA256
import hashlib
def hash_file(hash_file):
    sha256_hash = hashlib.sha256()
    with open(hash_file, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
print("SHA256 Hash of the file is: ")
print(hash_file("hashfile.txt"))
