import subprocess

# List of Python files to run
python_files = [
    #'/home/student/Desktop/trainModel.py',
    #'/home/student/Desktop/augmentData.py',
    #'/home/student/Desktop/trainModelWithAug.py',
    '/home/student/Desktop/createHCframes.py',
    '/home/student/Desktop/trainModelOnId.py',
    '/home/student/Desktop/createHCframes2.py',
    '/home/student/Desktop/trainModelOnId2.py',
    '/home/student/Desktop/createHCframesOOD.py',
    '/home/student/Desktop/trainModelOnOod.py',
]

for file in python_files:
    print(f"Running {file}...")
    result = subprocess.run(['python3', file], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"Error running {file}")
        break

print("All files have been run.")