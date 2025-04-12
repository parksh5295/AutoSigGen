import os

def select_csv_file():
    directory_path = '../Dataset/load_dataset/CICIDS2017/MachineLearningCSV'
    
    # Get a list of CSV files in a directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print("There are no CSV files to select.")
        return None

    # Output for users to select files
    print("Choose one of the following CSV files:")
    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file}")
    
    # 파일 선택 받기
    while True:
        try:
            choice = int(input(f"Enter the file number you want to select (1-{len(csv_files)}): "))
            if 1 <= choice <= len(csv_files):
                selected_file = csv_files[choice - 1]
                return os.path.join(directory_path, selected_file)
            else:
                print(f"1과 {len(csv_files)} Enter a number between."), choice  # return filenum
        except ValueError:
            print("Enter a valid number.")
