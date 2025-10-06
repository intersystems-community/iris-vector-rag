import os

import yaml

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_config():
    """
    Loads and parses the config.yaml file.

    Returns:
        dict: A dictionary containing the configuration parameters.
              Returns None if the file is not found or is malformed.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Configuration file not found at {CONFIG_FILE_PATH}")
        return None
    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {CONFIG_FILE_PATH}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {CONFIG_FILE_PATH}: {e}")
        return None


if __name__ == "__main__":
    # Example usage:
    config = get_config()
    if config:
        print("Configuration loaded successfully:")
        print(f"  Database Host: {config.get('database', {}).get('db_host')}")
        print(f"  Embedding Model: {config.get('embedding_model', {}).get('name')}")
        # Add more prints here to test other config values if needed
    else:
        print("Failed to load configuration.")
