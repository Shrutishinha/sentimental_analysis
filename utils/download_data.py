import os
import requests

def download_imdb_dataset():
    url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    output_dir = "data"
    output_path = os.path.join(output_dir, "IMDB_Dataset.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
            print(f"Dataset successfully downloaded and saved to: {output_path}")
            print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print(f"HTTP Error: Received status code {response.status_code}")
            return False
    except Exception as e:
        print(f"An error occurred while downloading: {e}")
        return False

if __name__ == "__main__":
    download_imdb_dataset()
