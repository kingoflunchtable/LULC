import os
import zipfile
import urllib.request

# 1. Download the dataset directly
url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
zip_path = "data.zip"

print("Downloading EuroSAT dataset (95MB)... This may take a minute.")
urllib.request.urlretrieve(url, zip_path)
print("Download complete!")

# 2. Extract the data
print("Extracting files...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".")
print("Extraction complete!")

# 3. Clean up
os.remove(zip_path)
print("Done!")   
