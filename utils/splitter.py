from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.loader import load_pdf
import re


documents = load_pdf("data/sample.pdf")


def is_heading(line, next_line=""):
    line = line.strip()

    if not line:
        return False
    if len(line.split()) > 5:        # headings are short (max 5 words)
        return False
    if line[0].islower():            # headings start with capital
        return False
    if line.endswith((".",":",",")):  # bullets/labels end with punctuation
        return False
    if next_line.startswith(" ") or next_line.startswith("-"):  # followed by bullets
        return False

    return True


def split_by_sections(documents):
    all_chunks = []

    for doc in documents:
        lines    = doc.page_content.split("\n")
        metadata = doc.metadata

        current_heading = "General"
        current_lines   = []

        for i, line in enumerate(lines):
            next_line = lines[i + 1] if i + 1 < len(lines) else ""

            if is_heading(line, next_line):
                # Save previous section
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        all_chunks.extend(
                            make_chunks(section_text, current_heading, metadata)
                        )
                current_heading = line.strip()
                current_lines   = [line]
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                all_chunks.extend(
                    make_chunks(section_text, current_heading, metadata)
                )

    return all_chunks


def make_chunks(section_text, heading, metadata):
    chunks = []

    if len(section_text) <= 800:
        enriched = (
            f"Document: {metadata.get('source', 'Unknown')}\n"
            f"Page: {metadata.get('page', '?')}\n"
            f"Section: {heading}\n\n"
            f"{section_text}"
        )
        chunks.append(Document(
            page_content=enriched,
            metadata={**metadata, "section": heading}
        ))
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=160,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        sub_chunks = splitter.split_text(section_text)

        for i, sub in enumerate(sub_chunks):
            label    = f"{heading} (part {i+1})" if len(sub_chunks) > 1 else heading
            enriched = (
                f"Document: {metadata.get('source', 'Unknown')}\n"
                f"Page: {metadata.get('page', '?')}\n"
                f"Section: {label}\n\n"
                f"{sub}"
            )
            chunks.append(Document(
                page_content=enriched,
                metadata={**metadata, "section": label}
            ))

    return chunks


def display_chunks(chunks, preview_length=300):
    print(f"Total Chunks: {len(chunks)}\n")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        content   = chunk.page_content
        last_char = content.strip()[-1]
        is_cut    = last_char not in ".!?"

        print(f"\nCHUNK {i + 1}")
        print(f"  Section  : {chunk.metadata.get('section', '?')}")
        print(f"  Length   : {len(content)} chars")
        print(f"  Preview  :\n{content[:preview_length]}")

        if is_cut:
            print("  ⚠️  Warning: Chunk may be cut mid-sentence")

        print("-" * 60)


chunks = split_by_sections(documents)
display_chunks(chunks)