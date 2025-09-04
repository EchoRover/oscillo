import re


def split_into_chunks(paragraph, max_chunk_len=300):
    sentences = re.split(r"(?<=[.!?]) +", paragraph)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_len:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def tag_textbook_structure(text):
    # Fix hyphenated words broken by line breaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Combine lines into paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Title lines with page numbers
    text = re.sub(
        r"^(\d+(\.\d+)*\s+.*?)(\s+\d{1,4})$",
        r"<TITLE> \1 <PAGE> \3 <END>",
        text,
        flags=re.MULTILINE,
    )

    # Section headers like Preface or Appendices
    text = re.sub(
        r"^(Preface|Index|Appendix|[A-Z]\.\d+\s+.*)$",
        r"<TITLE> \1 <END>",
        text,
        flags=re.MULTILINE,
    )

    # Split paragraphs by double newlines
    paragraphs = re.split(r"\n\s*\n", text)

    processed = []
    for para in paragraphs:
        chunks = split_into_chunks(para)
        for chunk in chunks:
            processed.append(f"<PARA> {chunk.strip()} <END>")

    return "\n\n".join(processed)


def filter_textbook_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        raw_text = infile.read()

    filtered_text = tag_textbook_structure(raw_text)

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(filtered_text)

    print(f"✅ Filtered file saved to: {output_path}")


with open("text.txt") as f:
    print(len(f.read()))
with open("filtered_physics.txt") as f:
    print(len(f.read()))

# Example usage:
if __name__ == "__mai":
    input_file = "text.txt"  # <-- your original textbook file
    output_file = "filtered_physics.txt"  # <-- will contain tagged/cleaned version
    filter_textbook_file(input_file, output_file)
