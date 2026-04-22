import gdown
import os

os.makedirs("app/models", exist_ok=True)

folder_id = "1_S2sUPfLECiTsj57FqCUxX4_G8PSpx_Z"
gdown.download_folder(
    f"https://drive.google.com/drive/folders/{folder_id}",
    output="app/models",
    quiet=False,
    use_cookies=False
)