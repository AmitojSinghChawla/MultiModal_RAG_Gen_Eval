"""
view_images.py
--------------
Display all images from chunks.json in your browser.

Shows each image with its chunk_id, source PDF, and LLaVA description
so you can easily find the chunk_id when writing questions.

Usage:
    python view_images.py

This creates images_gallery.html and opens it in your browser.
"""

import json
import base64
import webbrowser
import os

CHUNKS_FILE = "chunks.json"
OUTPUT_HTML = "images_gallery.html"


def load_image_chunks():
    """Load all chunks with modality='image' from chunks.json"""
    images = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            chunk = json.loads(line)
            if chunk.get("modality") == "image" and chunk.get("image_b64"):
                images.append(chunk)

    return images


def create_html_gallery(image_chunks):
    """Create an HTML page displaying all images"""

    html_parts = [
        """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Image Gallery - chunks.json</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .stats {
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .image-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .chunk-id {
            font-family: 'Courier New', monospace;
            background: #e3f2fd;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 14px;
            color: #1976d2;
            font-weight: bold;
        }
        .source-pdf {
            color: #666;
            font-size: 14px;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .description {
            background: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
            margin-top: 15px;
        }
        .description-label {
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 8px;
        }
        .description-text {
            color: #333;
            line-height: 1.6;
        }
        .copy-button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 10px;
        }
        .copy-button:hover {
            background: #45a049;
        }
        .copy-button:active {
            background: #3d8b40;
        }
    </style>
</head>
<body>
    <h1>📷 Image Gallery from chunks.json</h1>

    <div class="stats">
        <strong>Total images found:</strong> """
        + str(len(image_chunks))
        + """
    </div>
"""
    ]

    for i, chunk in enumerate(image_chunks, 1):
        chunk_id = chunk["chunk_id"]
        source_pdf = chunk["source_pdf"]
        description = chunk["retrieval_text"]
        image_b64 = chunk["image_b64"]

        html_parts.append(f"""
    <div class="image-card">
        <div class="image-header">
            <div>
                <span class="chunk-id" id="id-{i}">{chunk_id}</span>
                <button class="copy-button" onclick="copyToClipboard('id-{i}')">Copy ID</button>
            </div>
            <div class="source-pdf">📄 {source_pdf}</div>
        </div>

        <div class="image-container">
            <img src="data:image/jpeg;base64,{image_b64}" alt="Image {i}">
        </div>

        <div class="description">
            <div class="description-label">LLaVA Description:</div>
            <div class="description-text">{description}</div>
        </div>
    </div>
""")

    html_parts.append("""
    <script>
        function copyToClipboard(elementId) {
            const element = document.getElementById(elementId);
            const text = element.textContent;

            navigator.clipboard.writeText(text).then(() => {
                const button = element.nextElementSibling;
                const originalText = button.textContent;
                button.textContent = '✓ Copied!';
                button.style.background = '#2196F3';

                setTimeout(() => {
                    button.textContent = originalText;
                    button.style.background = '#4CAF50';
                }, 2000);
            });
        }
    </script>
</body>
</html>
""")

    return "".join(html_parts)


def main():
    print("Loading images from chunks.json...")

    if not os.path.exists(CHUNKS_FILE):
        print(f"ERROR: {CHUNKS_FILE} not found.")
        print("Run chunk_exporter.py first to create chunks.json")
        return

    image_chunks = load_image_chunks()

    if not image_chunks:
        print("No images found in chunks.json")
        print(
            "Make sure your PDFs contain images and chunk_exporter.py processed them."
        )
        return

    print(f"Found {len(image_chunks)} images")
    print(f"Creating HTML gallery...")

    html_content = create_html_gallery(image_chunks)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Gallery created: {OUTPUT_HTML}")
    print("Opening in browser...")

    # Open in default browser
    webbrowser.open(f"file://{os.path.abspath(OUTPUT_HTML)}")

    print("\nDone! You can now:")
    print("  1. See all images with their chunk_ids")
    print("  2. Click 'Copy ID' to copy the chunk_id for annotation")
    print("  3. Read the LLaVA description for each image")


if __name__ == "__main__":
    main()
