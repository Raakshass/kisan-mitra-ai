import os

print("Current working directory:", os.getcwd())
print("Listing current directory:")
print(os.listdir())

print("\nManual load attempt:")
with open(".env", "r") as f:
    for line in f:
        print(repr(line))
