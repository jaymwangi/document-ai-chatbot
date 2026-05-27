"""
PDF Loader Module - Task 2 of RAG Pipeline

Responsibility: Convert PDF files into clean, readable text strings.
Single responsibility: PDF → raw text. Nothing more.

This module handles:
- Opening PDF files safely
- Extracting text page by page
- Basic cleanup (no heavy preprocessing)
- Graceful error handling
"""

from pypdf import PdfReader
from pathlib import Path
import re
from typing import Optional, Tuple


class PDFLoadError(Exception):
    """Custom exception for PDF loading failures"""
    pass


def load_pdf(pdf_path: str, light_clean: bool = True) -> str:
    """
    Extract all text from a PDF file as a single string.
    
    Args:
        pdf_path: Path to the PDF file
        light_clean: If True, performs basic whitespace cleanup
    
    Returns:
        Combined text from all pages as a single string
    
    Raises:
        PDFLoadError: If file doesn't exist, can't be read, or has no text
    
    Example:
        >>> text = load_pdf("data/document.pdf")
        >>> print(len(text))
        12450
    """
    
    # Step 1: Validate file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise PDFLoadError(f"PDF file not found: {pdf_path}")
    
    if not pdf_file.suffix.lower() == '.pdf':
        raise PDFLoadError(f"File is not a PDF: {pdf_path}")
    
    # Step 2: Open PDF safely
    try:
        reader = PdfReader(pdf_file)
    except Exception as e:
        raise PDFLoadError(f"Cannot open PDF: {e}")
    
    # Step 3: Extract text page by page
    all_text = []
    total_pages = len(reader.pages)
    pages_with_text = 0
    
    for page_num, page in enumerate(reader.pages, 1):
        try:
            extracted = page.extract_text()
            if extracted and extracted.strip():
                all_text.append(extracted)
                pages_with_text += 1
            else:
                # Page has no extractable text (image, scan, or empty)
                all_text.append(f"[Page {page_num}: No text extracted]")
        except Exception as e:
            # Log but continue with other pages
            all_text.append(f"[Page {page_num}: Extraction error - {str(e)}]")
    
    # Step 4: Combine all pages
    raw_text = "\n\n".join(all_text)  # Double newline between pages
    
    # Step 5: Light cleanup (optional, deterministic only)
    if light_clean:
        raw_text = _light_cleanup(raw_text)
    
    # Step 6: Validate we got something
    if not raw_text.strip():
        raise PDFLoadError(
            f"No text extracted from {pdf_path}. "
            "This PDF may be scanned or image-based. "
            "OCR would be required for this document."
        )
    
    # Step 7: Return with metadata comment (not stored, just informative)
    return raw_text


def _light_cleanup(text: str) -> str:
    """
    Minimal, deterministic cleanup.
    Does NOT do: stopwords, lowercasing, stemming, or semantic changes.
    """
    # Replace 3+ newlines with 2 (preserves paragraph separation)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove carriage returns (Windows line endings)
    text = text.replace('\r\n', '\n')
    
    # Remove control characters except newline and tab
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Remove excessive spaces (multiple spaces → single space)
    text = re.sub(r' +', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def load_pdf_with_metadata(pdf_path: str) -> Tuple[str, dict]:
    """
    Extract text AND metadata from PDF.
    Useful for debugging or portfolio展示.
    
    Returns:
        (text, metadata_dict)
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise PDFLoadError(f"PDF file not found: {pdf_path}")
    
    reader = PdfReader(pdf_file)
    
    # Extract metadata
    metadata = {
        "filename": pdf_file.name,
        "file_size_kb": round(pdf_file.stat().st_size / 1024, 2),
        "num_pages": len(reader.pages),
    }
    
    # Add PDF internal metadata if available
    if reader.metadata:
        for key, value in reader.metadata.items():
            clean_key = key.replace('/', '').lower()
            metadata[clean_key] = value
    
    # Extract text
    text = load_pdf(pdf_path, light_clean=True)
    
    return text, metadata


def get_first_pdf(data_dir: str = "data") -> Optional[str]:
    """
    Convenience function: returns path to first PDF in data directory.
    
    Returns:
        Path string or None if no PDF found
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return None
    
    pdfs = list(data_path.glob("*.pdf"))
    if not pdfs:
        # Also check uppercase .PDF
        pdfs = list(data_path.glob("*.PDF"))
    
    return str(pdfs[0]) if pdfs else None


def validate_pdf(pdf_path: str) -> dict:
    """
    Check if a PDF is usable without full extraction.
    Useful for preprocessing validation.
    
    Returns:
        Dict with validation results
    """
    result = {
        "valid": False,
        "issues": [],
        "page_count": 0,
        "has_text": False,
        "is_scanned": False,
    }
    
    try:
        reader = PdfReader(pdf_path)
        result["page_count"] = len(reader.pages)
        
        # Check first 3 pages for text
        pages_with_text = 0
        for i in range(min(3, result["page_count"])):
            page_text = reader.pages[i].extract_text()
            if page_text and len(page_text.strip()) > 50:
                pages_with_text += 1
        
        if pages_with_text > 0:
            result["has_text"] = True
            result["valid"] = True
        else:
            result["is_scanned"] = True
            result["issues"].append("No text found - appears to be scanned/image PDF")
            
    except Exception as e:
        result["issues"].append(f"Error reading PDF: {str(e)}")
    
    return result


# ========== MAIN: Self-test when run directly ==========
if __name__ == "__main__":
    print("=" * 60)
    print("📄 PDF Loader - Task 2 Test")
    print("=" * 60)
    
    # Find a PDF to test with
    pdf_path = get_first_pdf()
    
    if not pdf_path:
        print("\n❌ No PDF found in 'data/' folder")
        print("   Please:")
        print("   1. Create a 'data' folder in the same directory")
        print("   2. Place a PDF file inside")
        print("   3. Run this script again")
        exit(1)
    
    print(f"\n📁 Testing PDF: {pdf_path}")
    
    # Validation first
    print("\n🔍 Running validation...")
    validation = validate_pdf(pdf_path)
    print(f"   Valid: {validation['valid']}")
    print(f"   Pages: {validation['page_count']}")
    print(f"   Has text: {validation['has_text']}")
    if validation['issues']:
        print(f"   Issues: {validation['issues']}")
    
    if not validation['valid']:
        print("\n❌ PDF validation failed. Cannot continue.")
        print("   (Scanned PDFs require OCR, which is out of scope for Task 2)")
        exit(1)
    
    # Load text
    print("\n📖 Loading text...")
    try:
        text = load_pdf(pdf_path)
        print(f"   ✅ Loaded {len(text)} characters")
        
        # Load with metadata
        text_with_meta, metadata = load_pdf_with_metadata(pdf_path)
        print(f"\n📊 Document Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        # Show samples
        print("\n📝 Text Preview (first 500 chars):")
        print("-" * 40)
        print(text[:500])
        print("-" * 40)
        
        print("\n📝 Text Preview (last 500 chars):")
        print("-" * 40)
        print(text[-500:])
        print("-" * 40)
        
        # Quality checks
        print("\n✅ Quality Verification:")
        
        # Check for page markers (our own insertion)
        if "[Page" in text:
            print("   ⚠️  Some pages had no extractable text (images/tables)")
        else:
            print("   ✅ All pages extracted successfully")
        
        # Check length is reasonable
        if len(text) > 1000:
            print("   ✅ Document length is substantial")
        else:
            print("   ⚠️  Document is very short (<1000 chars)")
        
        print("\n" + "=" * 60)
        print("✅ Task 2 Complete - PDF Loader is working!")
        print("=" * 60)
        
    except PDFLoadError as e:
        print(f"\n❌ PDF Load Error: {e}")
        exit(1)