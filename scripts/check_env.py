import faiss
import langchain
from langchain_mistralai import MistralAIEmbeddings
from mistralai.client import Mistral


def main() -> None:
    print(f"langchain: OK ({langchain.__version__})")
    print(f"faiss: OK ({faiss.__version__})")
    print(f"client mistralai: OK ({Mistral.__name__})")
    print(f"embeddings langchain_mistralai: OK ({MistralAIEmbeddings.__name__})")


if __name__ == "__main__":
    main()
