import re


def clean_text(file_path: str) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    start_match = re.search(
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text
    )
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    if start_match and end_match:
        text = text[start_match.end() : end_match.start()]
    text = text.upper()
    cleaned_text = re.sub(r"[^A-Z]" ," ", text)

    with open("book_cleaned.txt", "w", encoding="utf-8") as file_write:
        file_write.writelines(cleaned_text)


clean_text("book.txt")