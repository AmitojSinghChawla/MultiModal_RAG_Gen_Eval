"""
01chunk_exporter.py
-------------------
Reads PDFs from a directory, partitions them into text / table / image chunks,
and writes every chunk as a JSON line to chunks.json.

Fixes applied vs original:
  - try/except around partition_pdf so one bad PDF doesn't crash the whole run
  - Consistent use of utils.tokenize (not relevant here but imported for parity)
  - Minor: cleaned up comments for clarity

Usage:
    python 01chunk_exporter.py
"""

import json
import uuid
import os
from bs4 import BeautifulSoup
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv(verbose=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running the script.")



# ===============================
# 1. PDF Partition
# ===============================

def create_chunks_from_pdf(file_path):
    """
    Partition a PDF into structured elements using unstructured.

    strategy="hi_res"          — uses layout detection model for best accuracy
    extract_images_in_pdf=True — extracts image blocks
    extract_image_block_to_payload=True — puts base64 image data on the element
    chunking_strategy="by_title" — groups content by document sections/headings
    max_characters=10000       — hard cap per chunk
    combine_text_under_n_chars=2000 — merges tiny fragments
    new_after_n_chars=6000     — soft split target
    """
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=5000,
        combine_text_under_n_chars=3000,
        new_after_n_chars=1000,
    )
    return elements


# ===============================
# 2. Element Segregation
# ===============================

def table_text_segregation(all_elements):
    """
    Split elements into two lists: tables and text (CompositeElement) chunks.
    Tables need special HTML-to-text conversion.
    CompositeElements carry the main textual content and may embed images.
    """
    tables = []
    texts  = []
    for el in all_elements:
        if isinstance(el, Table):
            tables.append(el)
        elif isinstance(el, CompositeElement):
            texts.append(el)
    return tables, texts


def get_images(chunks):
    """
    Extract base64 images from CompositeElement chunks.

    Filters applied:
      - Skip images with no base64 data
      - Skip images smaller than ~1 KB (logos, icons, decorative elements)
        A base64 string of 1500 chars ≈ 1125 raw bytes
      - Skip duplicate images using first 300 chars as a fast fingerprint
    """
    images_b64  = []
    seen_hashes = set()

    for chunk in chunks:
        if not isinstance(chunk, CompositeElement):
            continue

        chunk_els = chunk.metadata.orig_elements or []
        for el in chunk_els:
            if "Image" not in str(type(el)):
                continue

            img_b64 = el.metadata.image_base64

            if not img_b64:
                continue

            # Filter: too small to be meaningful
            if len(img_b64) < 1500:
                continue

            # Filter: duplicate
            img_hash = img_b64[:300]
            if img_hash in seen_hashes:
                continue

            seen_hashes.add(img_hash)
            images_b64.append(img_b64)

    return images_b64


# ===============================
# 3. Text Normalization
# ===============================

def clean_text(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return " ".join(text.split())


def html_table_to_text(html: str) -> str:
    """
    Convert an HTML table string to a plain-text pipe-delimited representation.

    Example:
        <table>
          <tr><th>Name</th><th>Age</th></tr>
          <tr><td>Alice</td><td>30</td></tr>
        </table>
    becomes:
        Name | Age
        Alice | 30
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# ===============================
# 4. Image Description via OPEN AI
# ===============================

# GPT-4o is used here because it has the strongest vision capability.
# gpt-4o-mini is a cheaper fallback but produces noticeably vaguer descriptions
# for charts, diagrams, and tables-as-images — which hurts retrieval quality.

_vision_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0,          # deterministic — we want consistent descriptions
    max_tokens=1024,        # enough for a thorough description of complex figures
)

_parser = StrOutputParser()


def describe_image(image_b64: str) -> str:
    """
    Send a base64 image to GPT-4o vision and return a detailed text description
    suitable for both BM25 keyword indexing and dense semantic embedding.

    The prompt is carefully written to extract:
      - Chart/graph type and all axis labels, units, and data values
      - Table structure and cell contents (when rendered as an image)
      - Diagram components, relationships, and labels
      - Any visible numbers, percentages, dates, or named entities
      - Overall conclusion or trend visible in the figure

    These specifics are exactly what makes BM25 work (exact term overlap)
    and what makes dense embeddings semantically rich (meaningful content
    rather than generic visual descriptions like "a colorful bar chart").

    Parameters
    ----------
    image_b64 : str
        Raw base64-encoded image string (no data URI prefix needed here —
        we add it inside this function).

    Returns
    -------
    str
        A detailed plain-text description. Falls back to a placeholder
        string on API failure so ingestion continues without crashing.
    """
    data_url = f"data:image/jpeg;base64,{image_b64}"

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url"   : data_url,
                    "detail": "high",   # use high-res mode for charts with small text
                },
            },
            {
                "type": "text",
                "text": (
                    "You are a precise technical analyst. "
                    "Describe this image in as much detail as possible for a document retrieval system. "
                    "Your description will be used as a search index, so specificity is critical.\n\n"

                    "Cover ALL of the following that apply:\n"
                    "1. Type of visual (bar chart, line graph, pie chart, table, flowchart, architecture diagram, photograph, etc.)\n"
                    "2. Title or heading visible in the image\n"
                    "3. All axis labels, units of measurement, and scale ranges (for charts/graphs)\n"
                    "4. Every data series name and its approximate values at key points\n"
                    "5. All text visible in the image — labels, legends, annotations, captions\n"
                    "6. Any specific numbers, percentages, dates, currency values, or named entities\n"
                    "7. The overall trend, conclusion, or key insight the visual communicates\n"
                    "8. For tables: the column headers and a summary of the row data\n"
                    "9. For diagrams: the components, their relationships, and directional flow\n\n"

                    "Be specific. Do not say 'some values' — state the actual values. "
                    "Do not say 'various categories' — name the categories. "
                    "A researcher should be able to answer factual questions about this figure "
                    "using only your description."
                ),
            },
        ]
    )

    try:
        response = _vision_llm.invoke([message])
        return _parser.invoke(response)

    except Exception as e:
        print(f"   WARNING: GPT-4o vision failed — {e}")
        return "Image description unavailable due to API error."


# ===============================
# 5. Export helper
# ===============================

def export_chunk(record: dict, output_file: str) -> None:
    """Append one chunk record as a JSON line to the output file."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ===============================
# 6. Main Processing Loop
# ===============================

def process_pdfs_in_directory(directory_path: str, output_file: str = "chunks.json") -> None:
    """
    Iterate over every PDF in directory_path, partition each one, and write
    all chunks to output_file in JSON-lines format.

    Each PDF is wrapped in try/except so a single corrupt file doesn't abort
    the entire ingestion run.
    """

    # Clear output file so we never mix stale and fresh data
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Cleared old {output_file}")

    total_texts  = 0
    total_tables = 0
    total_images = 0

    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(directory_path, filename)
        print(f"\n📘 Processing: {filename}")

        # --- Partition PDF (with error recovery) ---
        try:
            elements = create_chunks_from_pdf(file_path)
        except Exception as e:
            print(f"   ERROR partitioning {filename}: {e} — skipping.")
            continue

        print(f"   Elements found: {len(elements)}")

        tables, texts = table_text_segregation(elements)
        images        = get_images(elements)

        print(f"   Texts: {len(texts)}, Tables: {len(tables)}, Images: {len(images)}")

        # -------- TEXT CHUNKS --------
        for text in texts:
            raw_text = text.text if hasattr(text, "text") else str(text)
            raw_text = clean_text(raw_text)

            record = {
                "chunk_id"      : str(uuid.uuid4()),
                "modality"      : "text",
                "source_pdf"    : filename,
                "page_number"   : getattr(text.metadata, "page_number", None),
                "raw_text"      : raw_text,
                "image_b64"     : None,
                "retrieval_text": raw_text,
            }
            export_chunk(record, output_file)

        # -------- TABLE CHUNKS --------
        for table in tables:
            if hasattr(table, "metadata") and hasattr(table.metadata, "text_as_html"):
                table_text = html_table_to_text(table.metadata.text_as_html)
            else:
                table_text = clean_text(str(table))

            record = {
                "chunk_id"      : str(uuid.uuid4()),
                "modality"      : "table",
                "source_pdf"    : filename,
                "page_number"   : getattr(table.metadata, "page_number", None),
                "raw_text"      : table_text,
                "image_b64"     : None,
                "retrieval_text": table_text,
            }
            export_chunk(record, output_file)

        # -------- IMAGE CHUNKS --------
        print(f"   Describing {len(images)} image(s) with GPT-4o...")
        for i, image in enumerate(images):
            print(f"   Image {i+1}/{len(images)}...", end=" ", flush=True)
            description = describe_image(image)
            print("done")

            record = {
                "chunk_id"      : str(uuid.uuid4()),
                "modality"      : "image",
                "source_pdf"    : filename,
                "page_number"   : None,
                "raw_text"      : None,
                "image_b64"     : image,
                "retrieval_text": description,
            }
            export_chunk(record, output_file)

        total_texts  += len(texts)
        total_tables += len(tables)
        total_images += len(images)

        print(f"   ✅ Done: {filename}")

    print(f"\n{'='*40}")
    print(f"INGESTION COMPLETE")
    print(f"  Text chunks  : {total_texts}")
    print(f"  Table chunks : {total_tables}")
    print(f"  Image chunks : {total_images}")
    print(f"  Output       : {output_file}")
    print(f"{'='*40}")


# ===============================
# 7. Entry Point
# ===============================

if __name__ == "__main__":
    pdf_directory = r"D:\Projects\Major_Project\documents"
    process_pdfs_in_directory(pdf_directory)