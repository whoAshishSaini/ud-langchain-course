import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from ud-langchain-course!")
    print(os.environ.get("GROQ_API_KEY"))


if __name__ == "__main__":
    main()
