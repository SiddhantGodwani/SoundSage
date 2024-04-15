import os

# Set the directory where the Lottie animation file is located
lottie_dir = "C:\Users\SIDDHANT GODWANI\Desktop\Streamlit Music R\Loading_Animation.json"

# Get a list of files in the directory
files = os.listdir(lottie_dir)

# Find the Lottie animation JSON file
lottie_file = None
for file in files:
    if file.endswith(".json"):
        lottie_file = file
        break

if lottie_file:
    # Get the absolute path of the Lottie animation JSON file
    lottie_file_path = os.path.abspath(os.path.join(lottie_dir, lottie_file))
    print(f"Lottie animation file path: {lottie_file_path}")
else:
    print("Lottie animation JSON file not found.")