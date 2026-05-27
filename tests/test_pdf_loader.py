"""
Test Module: PDF Loader

Validates PDF loading functionality:
- File detection
- Validation logic
- Text extraction quality
"""

from core.pdf_loader import load_pdf, get_first_pdf, validate_pdf


def test_pdf_exists():
    """Check that a PDF file exists in data/ folder."""
    pdf_path = get_first_pdf()

    assert pdf_path is not None, "No PDF found in data/ folder"


def test_pdf_validation():
    """Check that PDF passes validation rules."""
    pdf_path = get_first_pdf()
    v = validate_pdf(pdf_path)

    assert v["valid"] is True, f"PDF not valid: {v['issues']}"


def test_pdf_text_extraction():
    """Check that text is extracted properly from PDF."""
    pdf_path = get_first_pdf()
    text = load_pdf(pdf_path)

    assert isinstance(text, str), "Output is not a string"
    assert len(text) > 0, "Extracted text is empty"
    assert len(text) > 500, "Extracted text is too small"


def run_tests():
    """Simple manual test runner."""
    tests = [
        test_pdf_exists,
        test_pdf_validation,
        test_pdf_text_extraction,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} → {e}")

    print(f"\nResult: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 PDF Loader is working correctly")


if __name__ == "__main__":
    run_tests()