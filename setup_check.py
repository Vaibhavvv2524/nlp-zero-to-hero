import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

RESULTS = []


def record(label, ok, detail=""):
    RESULTS.append((label, ok))
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def check_python_and_venv():
    info = sys.version_info
    record("Python version", info >= (3, 10), f"{info.major}.{info.minor}.{info.micro}")

    executable = Path(sys.executable)
    record("Python executable", True, str(executable))

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    points_to_venv = executable.resolve() == VENV_PYTHON.resolve()
    record(
        "Virtual environment activation",
        in_venv and points_to_venv,
        f"prefix={sys.prefix}",
    )


def check_import(label, module_name, dist_name=None):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        record(f"{label} import", False, f"{type(exc).__name__}: {exc}")
        return None

    version = getattr(module, "__version__", None)
    if not version and dist_name:
        try:
            from importlib import metadata

            version = metadata.version(dist_name)
        except Exception:
            version = "unknown"
    record(f"{label} import", True, str(version or "imported"))
    return module


def check_imports():
    check_import("NumPy", "numpy")
    check_import("Pandas", "pandas")
    check_import("NLTK", "nltk")
    check_import("Scikit-learn", "sklearn", "scikit-learn")
    check_import("Matplotlib", "matplotlib")
    check_import("Seaborn", "seaborn")
    check_import("SciPy", "scipy")
    check_import("spaCy", "spacy")
    check_import("regex", "regex")
    check_import("emoji", "emoji")
    check_import("BeautifulSoup", "bs4", "beautifulsoup4")
    check_import("ftfy", "ftfy")
    check_import("Langdetect", "langdetect", "langdetect")
    check_import("Wordcloud", "wordcloud")
    check_import("Jupyter", "jupyter_core", "jupyter-core")
    check_import("IPykernel", "ipykernel")


def check_nltk_resources():
    import nltk

    resources = {
        "punkt_tab": ["tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"],
        "stopwords": ["corpora/stopwords", "corpora/stopwords.zip"],
        "wordnet": ["corpora/wordnet", "corpora/wordnet.zip"],
        "omw-1.4": ["corpora/omw-1.4", "corpora/omw-1.4.zip"],
    }

    for name, candidates in resources.items():
        found = None
        for candidate in candidates:
            try:
                found = nltk.data.find(candidate)
                break
            except LookupError:
                continue
        record(f"NLTK resource '{name}'", found is not None, str(found or "missing"))


def check_nltk_tokenization():
    try:
        from nltk.corpus import stopwords
        from nltk.corpus import wordnet
        from nltk.tokenize import sent_tokenize, word_tokenize
    except Exception as exc:
        record("NLTK tokenization setup", False, f"{type(exc).__name__}: {exc}")
        return

    text = "I absolutely love learning Natural Language Processing!"
    expected_words = [
        "I",
        "absolutely",
        "love",
        "learning",
        "Natural",
        "Language",
        "Processing",
        "!",
    ]
    try:
        words = word_tokenize(text)
        record("NLTK word_tokenize()", words == expected_words, str(words))
    except Exception as exc:
        record("NLTK word_tokenize()", False, f"{type(exc).__name__}: {exc}")

    try:
        sentences = sent_tokenize("I love NLP. It is fun!")
        record(
            "NLTK sent_tokenize()",
            sentences == ["I love NLP.", "It is fun!"],
            str(sentences),
        )
    except Exception as exc:
        record("NLTK sent_tokenize()", False, f"{type(exc).__name__}: {exc}")

    try:
        english = stopwords.words("english")
        record("English stopwords", "the" in english and len(english) > 0, f"{len(english)} words")
    except Exception as exc:
        record("English stopwords", False, f"{type(exc).__name__}: {exc}")

    try:
        synsets = wordnet.synsets("dog")
        record("WordNet availability", len(synsets) > 0, f"{len(synsets)} synsets for 'dog'")
    except Exception as exc:
        record("WordNet availability", False, f"{type(exc).__name__}: {exc}")


def check_jupyter_kernel():
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().find_kernel_specs()
        ok = "nlp-zero-to-hero" in specs
        record(
            "Jupyter kernel 'nlp-zero-to-hero'",
            ok,
            specs.get("nlp-zero-to-hero", "not registered"),
        )
    except Exception as exc:
        record("Jupyter kernel 'nlp-zero-to-hero'", False, f"{type(exc).__name__}: {exc}")


def main():
    print("=" * 60)
    print("NLP Zero to Hero - Environment Verification")
    print("=" * 60)

    check_python_and_venv()
    check_imports()
    check_nltk_resources()
    check_nltk_tokenization()
    check_jupyter_kernel()

    passed = sum(1 for _, ok in RESULTS if ok)
    failed = len(RESULTS) - passed

    print("=" * 60)
    if failed == 0:
        print(f"OVERALL STATUS: PASS ({passed}/{len(RESULTS)} checks passed)")
        return 0

    print(f"OVERALL STATUS: FAIL ({passed}/{len(RESULTS)} passed, {failed} failed)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
