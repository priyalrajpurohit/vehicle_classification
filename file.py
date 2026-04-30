import os

structure = {
    "vehicle_type_classification": {
        "data": {
            "raw": {},
            "processed": {},
            "external": {}
        },
        "notebooks": {
            "01_data_exploration.ipynb": None,
            "02_feature_engineering.ipynb": None,
            "03_model_training.ipynb": None,
            "04_model_evaluation.ipynb": None
        },
        "src": {
            "__init__.py": None,
            "config": {
                "config.yaml": None
            },
            "data": {
                "__init__.py": None,
                "data_loader.py": None,
                "data_preprocessing.py": None
            },
            "features": {
                "__init__.py": None,
                "feature_builder.py": None
            },
            "models": {
                "__init__.py": None,
                "train.py": None,
                "evaluate.py": None,
                "predict.py": None
            }
        },
        "models": {
            ".gitkeep": None
        },
        "tests": {
            "__init__.py": None,
            "test_data.py": None,
            "test_features.py": None,
            "test_models.py": None
        },
        "app.py": None,
        "requirements.txt": None,
        "README.md": None,
        ".gitignore": None,
        "setup.py": None,
        "LICENSE": None
    }
}

def create_structure(base, tree):
    for name, content in tree.items():
        path = os.path.join(base, name)
        if content is None:
            # Create file
            os.makedirs(base, exist_ok=True)
            open(path, "w").close()
        else:
            # Create directory
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)

if __name__ == "__main__":
    create_structure(".", structure)
