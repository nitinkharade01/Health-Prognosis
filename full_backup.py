import os
import zipfile
import datetime

def backup_project(project_dir, backup_dir, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ['venv', '__pycache__']

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"{os.path.basename(project_dir)}_backup_{now}.zip"
    backup_path = os.path.join(backup_dir, backup_filename)

    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
        for foldername, subfolders, filenames in os.walk(project_dir):
            # Skip excluded folders
            if any(excluded in foldername for excluded in exclude_dirs):
                continue
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, start=project_dir)
                backup_zip.write(file_path, arcname)

    print(f"✅ Backup created: {backup_path}")
    return backup_path

if __name__ == "__main__":
    # Modify these paths as needed
    PROJECT_DIR = r"E:\Project\Health-Prognosis"
    BACKUP_DIR = r"E:\Project\Backups"

    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_project(PROJECT_DIR, BACKUP_DIR)
