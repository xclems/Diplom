filename = "book_cleaned.txt"
with open(filename, "r", encoding="utf-8") as file:
    content = file.read()
words = content.split()
word = max(words, key=len)
print(f"Slowo: {word}")
print(f"Symbolow: {len(word)}")