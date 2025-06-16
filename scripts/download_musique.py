import os
import subprocess
import urllib.request
import zipfile

def main():
    data_dir = os.path.join(os.getcwd(), "data", "musique")
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "musique_v1.0.zip")
    if not os.path.exists(zip_path):
        print("🔽 正在下载 MuSiQue 数据集 ZIP 文件（由代理服务器托管）...")
        url = "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_v1.0.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("✅ 下载完成:", zip_path)
    else:
        print("✓ ZIP 文件已存在")

    print("📦 正在解压 ZIP 文件...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("✅ 解压完成，文件已放入:", data_dir)

if __name__ == "__main__":
    main()
