import os
import argparse
import subprocess

def download_data(competition, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Downloading data from Kaggle competition: {competition}")

    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", competition,
        "-p", output_dir
    ], check=True)

    # Descomprimir si el archivo es ZIP
    for file in os.listdir(output_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(output_dir, file)
            subprocess.run(["unzip", "-o", zip_path, "-d", output_dir], check=True)
            os.remove(zip_path)

    print("[INFO] Download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    download_data(args.competition, args.output)
