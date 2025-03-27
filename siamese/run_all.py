import subprocess
import sys

def run_script(script_name):
    print(f"\n🚀 Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f" Error running {script_name}:")
        print(result.stderr)
        sys.exit(result.returncode)

if __name__ == "__main__":
    run_script("clean_database.py")
    run_script("data_loader.py")
    run_script("generate_embeddings.py")
    run_script("train_model.py")
    print("\n All scripts ran successfully!")
