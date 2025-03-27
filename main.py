from DataPreprocessor import DataPreprocessor


dirs_to_process = [
    "data/Test/Fake/",
    "data/Test/Real/",
    "data/Train/Fake/",
    "data/Train/Real/",
    "data/Validation/Fake",
    "data/ValidationReal",
]

for dir in dirs_to_process:
    print("Processing: ", dir)
    processor = DataPreprocessor(dir, dir.replace("data", "data_processed"))
    processor.process_directory()