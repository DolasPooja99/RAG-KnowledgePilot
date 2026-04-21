import re
import json

CHAPTER_PATTERNS = [
    r'^\s*chapter\s+(\d+|[ivxlcdmIVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten)\b',
    r'^\s*part\s+(\d+|[ivxlcdmIVXLCDM]+|one|two|three|four|five)\b',
    r'^\s*(\d{1,2})\s+[A-Z][A-Za-z\s]{3,60}$',  # "1  Introduction to..."
]

def detect_book_and_chapters(pages):
    """
    Returns (is_book, chapters).
    chapters = [{"name": str, "start_page": int, "end_page": int}]
    A PDF is considered a book only if 2+ chapter headings are found.
    """
    chapters = []

    for page_idx, page in enumerate(pages):
        for line in page.page_content.split('\n'):
            stripped = line.strip()
            if not stripped or len(stripped) > 120:
                continue
            for pattern in CHAPTER_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    # Avoid duplicate adjacent chapter headings from repeated TOC lines
                    if chapters and chapters[-1]['start_page'] == page_idx:
                        break
                    chapters.append({
                        'name': stripped[:100],
                        'start_page': page_idx,
                        'end_page': page_idx,
                    })
                    break

    for i in range(len(chapters) - 1):
        chapters[i]['end_page'] = chapters[i + 1]['start_page'] - 1
    if chapters:
        chapters[-1]['end_page'] = len(pages) - 1

    is_book = len(chapters) >= 2
    return is_book, (chapters if is_book else [])


def generate_flashcards(pages, chapter, num_cards, client):
    """
    Generate Q&A flashcards for a chapter using Claude.
    Returns list of {"question": str, "answer": str}.
    """
    start = chapter['start_page']
    # Cap pages sent to Claude to avoid token overflow
    end = min(chapter['end_page'], start + 25)
    chapter_text = '\n\n'.join(p.page_content for p in pages[start:end + 1])

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system="You are a study assistant that creates concise, accurate flashcards.",
        messages=[{
            "role": "user",
            "content": (
                f"Generate exactly {num_cards} flashcards from the text below.\n"
                "Return ONLY a JSON array — no explanation, no markdown fences.\n"
                'Format: [{"question": "...", "answer": "..."}, ...]\n\n'
                f"Text:\n{chapter_text[:8000]}"
            )
        }]
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if Claude adds them
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)
