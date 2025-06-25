import os

from dotenv import load_dotenv


def load_keys():
    load_dotenv()
    return {
        "api_key": os.getenv("API_KEY"),
        "secret_key": os.getenv("API_SECRET")
    }