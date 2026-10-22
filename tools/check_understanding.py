#!/usr/bin/env python3
"""Validation tool for docs/UNDERSTANDING.md human authorship gate."""

import re
import sys
from pathlib import Path


def check_scaffold(content: str) -> bool:
    """Validate that scaffold exists with all 12 questions and human placeholders."""
    for q_idx in range(1, 13):
        pattern = rf"### Q{q_idx}\b"
        if not re.search(pattern, content):
            print(f"[FAIL] Missing heading for Q{q_idx}")
            return False
    # Ensure at least 12 placeholder tags exist
    placeholders = re.findall(r"<!--\s*HUMAN:", content)
    if len(placeholders) < 12:
        print(f"[FAIL] Expected at least 12 <!-- HUMAN placeholders, found {len(placeholders)}")
        return False
    print("[PASS] docs/UNDERSTANDING.md scaffold is valid with all 12 questions.")
    return True


def check_completed(content: str) -> bool:
    """Validate that human has filled all questions and completed frontmatter."""
    if "<filled by human>" in content or "<human states" in content or "<YYYY-MM-DD>" in content:
        print("[FAIL] Front-matter contains uncompleted placeholders.")
        return False

    if "<!-- HUMAN" in content:
        print("[FAIL] File still contains unfilled '<!-- HUMAN' placeholders.")
        return False

    # Check questions
    sections = re.split(r"### Q\d+", content)
    # The first element is before Q1
    q_texts = sections[1:]
    if len(q_texts) < 10:
        print(f"[FAIL] Expected at least 10 question sections, found {len(q_texts)}")
        return False

    total_words = 0
    for idx, text in enumerate(q_texts, start=1):
        # strip markdown comments and non-text
        clean_text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        words = clean_text.strip().split()
        total_words += len(words)
        if len(words) < 40:
            print(f"[FAIL] Q{idx} has only {len(words)} words (minimum required: 40).")
            return False

    if total_words < 600:
        print(f"[FAIL] Total words is {total_words} (minimum required: 600).")
        return False

    print(
        f"[PASS] Human understanding gate passed with {total_words} words across {len(q_texts)} questions."
    )
    return True


def main():
    scaffold_only = "--scaffold-only" in sys.argv
    path = Path("docs/UNDERSTANDING.md")
    if not path.exists():
        print(f"[FAIL] {path} does not exist.")
        sys.exit(1)

    content = path.read_text(encoding="utf-8")
    if scaffold_only:
        if not check_scaffold(content):
            sys.exit(1)
        sys.exit(0)

    # In standard mode, if placeholders still exist, report informative failure
    if not check_completed(content):
        print(
            "\nNote: The 'understanding-gate' checks that the human author has personally filled out\n"
            "docs/UNDERSTANDING.md in their own words. As mandated by Rule R1, the AI agent is forbidden\n"
            "from drafting or filling these answers. Once the human author writes their answers, this gate will pass."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
