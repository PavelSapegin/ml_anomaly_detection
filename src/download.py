import os
import shutil

import kagglehub


def main() -> None:
    # Download latest version
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    target_dir = "./data"
    os.makedirs(target_dir, exist_ok=True)

    for file_name in os.listdir(path):
        full_file_name = os.path.join(path, file_name)

        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, target_dir)

    print(f"Files copied to {os.path.abspath(target_dir)}")


if __name__ == "__main__":
    main()
