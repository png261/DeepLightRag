"""
DeepSeek-OCR Output Parser
Parses and cleans the structured output from DeepSeek-OCR
Extracts clean text, bounding boxes, and document structure
"""

import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ParsedTextSegment:
    """A parsed text segment with its metadata"""
    text: str
    bbox: Optional[List[int]] = None
    confidence: float = 1.0
    segment_type: str = "text"  # text, reference, detection


class DeepSeekOutputParser:
    """
    Parser for DeepSeek-OCR structured output format.

    DeepSeek-OCR returns text with special tokens:
    - <|ref|>Text<|/ref|> - Main text content
    - <|det|>[[x1, y1, x2, y2]]<|/det|> - Bounding box coordinates
    """

    def __init__(self):
        # Compile regex patterns for efficiency
        self.ref_pattern = re.compile(r'<\|ref\|>(.*?)<\|/ref\|>', re.DOTALL)
        self.det_pattern = re.compile(r'<\|det\|>\[\[(.*?)\]\]<\|/det\|>')
        self.bbox_pattern = re.compile(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]')

    def parse_output(self, raw_output: str) -> List[ParsedTextSegment]:
        """
        Parse raw DeepSeek-OCR output into clean text segments

        Args:
            raw_output: Raw string output from DeepSeek-OCR

        Returns:
            List of parsed text segments with metadata
        """
        if not raw_output:
            return []

        segments = []

        # Extract all reference text segments
        ref_matches = self.ref_pattern.findall(raw_output)

        # Extract all detection boxes
        det_matches = self.det_pattern.findall(raw_output)

        # Parse bounding boxes
        bboxes = []
        for det in det_matches:
            bbox_match = self.bbox_pattern.search(det)
            if bbox_match:
                bbox = [int(x) for x in bbox_match.groups()]
                bboxes.append(bbox)

        # Create segments - each ref text can have a corresponding bbox
        for i, ref_text in enumerate(ref_matches):
            # Clean the text
            clean_text = self._clean_text(ref_text)

            if not clean_text.strip():
                continue

            # Get bbox if available
            bbox = bboxes[i] if i < len(bboxes) else None

            # Determine segment type
            segment_type = self._classify_segment(clean_text)

            segment = ParsedTextSegment(
                text=clean_text,
                bbox=bbox,
                confidence=1.0,  # DeepSeek-OCR doesn't provide per-segment confidence
                segment_type=segment_type
            )
            segments.append(segment)

        return segments

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Fix common OCR artifacts
        replacements = {
            '``': '"',
            "''": "'",
            '|': 'I',  # Common in OCR where I becomes |
            'l': 'l',   # Keep as is (already correct)
            '0': 'O',   # Sometimes confusing in OCR
        }

        # Only replace obvious artifacts
        for old, new in replacements.items():
            if old == '``' or old == "''":
                text = text.replace(old, new)

        return text

    def _classify_segment(self, text: str) -> str:
        """Classify the type of text segment"""
        text_lower = text.lower()

        # Check for headers
        if len(text) < 50 and text.isupper():
            return "header"

        # Check for list items
        if text_lower.startswith(('•', '-', '*')) or re.match(r'^\d+\.', text):
            return "list"

        # Check for table content
        if '\t' in text or re.search(r'\w+\s+\w+\s+\w+', text):
            # Multiple words spaced out could be table
            words = text.split()
            if len(words) >= 3 and all(len(w) < 20 for w in words):
                return "table"

        # Check for figure caption
        if any(keyword in text_lower for keyword in ['figure', 'fig.', 'image', 'photo']):
            return "caption"

        # Check for block quote
        if text.startswith('"') and text.endswith('"'):
            return "quote"

        # Default to paragraph
        return "paragraph"

    def merge_segments(self, segments: List[ParsedTextSegment],
                       merge_same_type: bool = True,
                       max_merge_distance: int = 50) -> List[ParsedTextSegment]:
        """
        Merge consecutive segments of the same type

        Args:
            segments: List of parsed segments
            merge_same_type: Whether to merge segments of same type
            max_merge_distance: Max character distance to merge

        Returns:
            List of merged segments
        """
        if not segments:
            return []

        merged = [segments[0]]

        for current in segments[1:]:
            last = merged[-1]

            # Check if we should merge
            should_merge = False

            if merge_same_type and last.segment_type == current.segment_type:
                should_merge = True

            # Also check if they're visually close (if bboxes exist)
            if last.bbox and current.bbox:
                y_distance = abs(last.bbox[1] - current.bbox[1])
                if y_distance < max_merge_distance:
                    should_merge = True

            if should_merge:
                # Merge with last segment
                last.text = last.text + " " + current.text

                # Update bbox to encompass both
                if last.bbox and current.bbox:
                    x1 = min(last.bbox[0], current.bbox[0])
                    y1 = min(last.bbox[1], current.bbox[1])
                    x2 = max(last.bbox[2], current.bbox[2])
                    y2 = max(last.bbox[3], current.bbox[3])
                    last.bbox = [x1, y1, x2, y2]

                # Average confidence
                last.confidence = (last.confidence + current.confidence) / 2
            else:
                merged.append(current)

        return merged

    def extract_structure(self, segments: List[ParsedTextSegment]) -> Dict[str, Any]:
        """
        Extract document structure from segments

        Args:
            segments: List of parsed segments

        Returns:
            Dictionary with structure information
        """
        structure = {
            "headers": [],
            "paragraphs": [],
            "lists": [],
            "tables": [],
            "figures": [],
            "quotes": [],
            "metadata": {}
        }

        for segment in segments:
            item = {
                "text": segment.text,
                "bbox": segment.bbox,
                "confidence": segment.confidence
            }

            if segment.segment_type == "header":
                structure["headers"].append(item)
            elif segment.segment_type == "paragraph":
                structure["paragraphs"].append(item)
            elif segment.segment_type == "list":
                structure["lists"].append(item)
            elif segment.segment_type == "table":
                structure["tables"].append(item)
            elif segment.segment_type == "caption":
                structure["figures"].append(item)
            elif segment.segment_type == "quote":
                structure["quotes"].append(item)

        # Extract metadata
        structure["metadata"] = {
            "total_segments": len(segments),
            "has_structure": any([
                structure["headers"],
                structure["lists"],
                structure["tables"],
                structure["quotes"]
            ]),
            "average_confidence": np.mean([s.confidence for s in segments]) if segments else 0.0
        }

        return structure

    def clean_and_format(self, raw_output: str,
                         preserve_structure: bool = True,
                         merge_paragraphs: bool = True) -> str:
        """
        Clean and format the raw output into readable text

        Args:
            raw_output: Raw string from DeepSeek-OCR
            preserve_structure: Whether to preserve document structure
            merge_paragraphs: Whether to merge short paragraphs

        Returns:
            Cleaned and formatted text
        """
        # Parse the output
        segments = self.parse_output(raw_output)

        if not segments:
            return ""

        # Optionally merge segments
        if merge_paragraphs:
            segments = self.merge_segments(segments)

        # Format based on structure
        if preserve_structure:
            lines = []
            current_indent = 0

            for segment in segments:
                text = segment.text

                if segment.segment_type == "header":
                    # Headers are typically already in caps or formatted
                    lines.append(f"\n{text}\n")
                elif segment.segment_type == "list":
                    # Add list marker if not present
                    if not re.match(r'^\s*[-•*]\s*', text):
                        text = f"• {text}"
                    lines.append(f"  {text}")
                elif segment.segment_type == "quote":
                    lines.append(f"> {text}")
                elif segment.segment_type == "paragraph":
                    lines.append(text)
                else:
                    lines.append(text)

            return "\n".join(lines)
        else:
            # Just join all text
            return " ".join(s.text for s in segments)