import os
from utils.extract_information import high_level_prompt, content_prompt, extract_from_pdf
import argparse

os.makedirs("./extracted", exist_ok=True)
os.makedirs("./extracted/GỢI Ý", exist_ok=True)
os.makedirs("./extracted/TÁC HẠI", exist_ok=True)
os.makedirs("./extracted/YẾU TỐ", exist_ok=True)
os.makedirs("./extracted/YẾU TỐ NGOẠI CẢNH", exist_ok=True)
os.makedirs("./extracted/SỨC KHOẺ", exist_ok=True)

parser = argparse.ArgumentParser(description="Extracting PDF files")
parser.add_argument('--dir', default='Database/YẾU TỐ/SỨC KHOẺ', help='directory to extract pdf files')
args = parser.parse_args()

if __name__ == "__main__":
    for path in os.listdir(args.dir)[255:]:
        print(path)
        path = os.path.join(args.dir, path)
        extract_from_pdf(path, content_prompt=content_prompt, high_level_prompt=high_level_prompt)