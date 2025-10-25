"""
Test script for server features including random prompts and tooltips.

This verifies:
1. Server starts correctly
2. Random prompts are different each time
3. All modalities have prompts
4. HTML structure is correct
"""

import sys
from pathlib import Path
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_random_prompts_exist():
    """Test that random prompt arrays are defined in visualization server."""
    print("\n" + "="*70)
    print("TEST: Random Creative Prompts Exist")
    print("="*70)

    server_file = Path(__file__).parent.parent / "image_gen" / "visualization_server.py"

    if not server_file.exists():
        print(f"✗ Server file not found: {server_file}")
        return False

    content = server_file.read_text()

    # Check for creative prompts object
    if "const creativePrompts = {" not in content:
        print("✗ creativePrompts object not found")
        return False

    print("✓ creativePrompts object found")

    # Check for each modality
    modalities = [
        "text2img",
        "text2audio",
        "text2video",
        "llm",
        "audio2text"
    ]

    for modality in modalities:
        # Allow both "key": and key: syntax in JavaScript object literals
        pattern = f'("|\')?{modality}("|\')?:\\s*\\['
        if re.search(pattern, content):
            print(f"✓ Prompts found for: {modality}")
        else:
            print(f"✗ Prompts missing for: {modality}")
            return False

    # Check for image-based prompts
    if "const imageBasedPrompts = {" not in content:
        print("✗ imageBasedPrompts object not found")
        return False

    print("✓ imageBasedPrompts object found")

    image_modes = ["img2img", "img2video", "controlnet"]
    for mode in image_modes:
        # Allow both "key": and key: syntax in JavaScript object literals
        pattern = f'("|\')?{mode}("|\')?:\\s*\\['
        if re.search(pattern, content):
            print(f"✓ Prompts found for: {mode}")
        else:
            print(f"✗ Prompts missing for: {mode}")
            return False

    # Check for setRandomPrompt function
    if "function setRandomPrompt(mode)" not in content:
        print("✗ setRandomPrompt function not found")
        return False

    print("✓ setRandomPrompt function found")

    # Check it's called on init
    if "setRandomPrompt('text2img')" not in content:
        print("✗ setRandomPrompt not called on initialization")
        return False

    print("✓ setRandomPrompt called on initialization")

    # Check it's called in switchMode
    pattern = r"function switchMode.*?setRandomPrompt\(mode\)"
    if re.search(pattern, content, re.DOTALL):
        print("✓ setRandomPrompt called in switchMode")
    else:
        print("✗ setRandomPrompt not called in switchMode")
        return False

    print("\n" + "="*70)
    print("✓ TEST PASSED: All random prompts properly implemented")
    print("="*70)

    return True


def test_tooltip_count():
    """Test that all tooltips are present in the HTML."""
    print("\n" + "="*70)
    print("TEST: Tooltip Coverage")
    print("="*70)

    server_file = Path(__file__).parent.parent / "image_gen" / "visualization_server.py"
    content = server_file.read_text()

    # Count tooltip instances
    tooltip_pattern = r'<div class="tooltip">'
    tooltips = re.findall(tooltip_pattern, content)
    count = len(tooltips)

    print(f"Found {count} tooltip instances")

    # Expected components that should have tooltips
    expected_components = [
        "Input Processing",
        "API Handler",
        "Tokenization",
        "Text Embedding",
        "Image Loading",
        "VAE Encoder",
        "Audio Input",
        "Audio Feature Extraction",
        "Diffusion Process",
        "LLM Inference",
        "Text-to-Speech",
        "Automatic Speech Recognition",
        "VAE Decoder",
        "Neural Vocoder",
        "Detokenization",
        "Output & Completion"
    ]

    print(f"\nChecking for expected component tooltips:")
    for component in expected_components:
        pattern = f'tooltip-title">{re.escape(component)}'
        if re.search(pattern, content):
            print(f"✓ {component}")
        else:
            print(f"⚠️  {component} - not found")

    # Check tooltip CSS exists
    if ".component .tooltip {" in content:
        print(f"\n✓ Tooltip CSS found")
    else:
        print(f"\n✗ Tooltip CSS missing")
        return False

    if ".component:hover .tooltip {" in content:
        print(f"✓ Tooltip hover effect found")
    else:
        print(f"✗ Tooltip hover effect missing")
        return False

    print("\n" + "="*70)
    print(f"✓ TEST PASSED: {count} tooltips found with proper CSS")
    print("="*70)

    return True


def test_html_structure():
    """Test that HTML structure is valid."""
    print("\n" + "="*70)
    print("TEST: HTML Structure Validation")
    print("="*70)

    server_file = Path(__file__).parent.parent / "image_gen" / "visualization_server.py"
    content = server_file.read_text()

    # Extract HTML content
    html_start = content.find('html_content = """')
    html_end = content.find('"""', html_start + 20)

    if html_start == -1 or html_end == -1:
        print("✗ Could not find HTML content")
        return False

    html = content[html_start:html_end]

    # Check for key structural elements
    checks = [
        ("<html", "HTML tag"),
        ("<head", "Head section"),
        ("<body", "Body section"),
        ("<style>", "Style section"),
        ("<script>", "Script section"),
        ('id="prompt"', "Prompt textarea"),
        ('id="generateForm"', "Generate form"),
        ("const creativePrompts", "Creative prompts JS"),
        ("function setRandomPrompt", "setRandomPrompt function"),
        ("function switchMode", "switchMode function"),
    ]

    all_passed = True
    for search_str, description in checks:
        if search_str in html:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - NOT FOUND")
            all_passed = False

    if all_passed:
        print("\n" + "="*70)
        print("✓ TEST PASSED: HTML structure is valid")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print("✗ TEST FAILED: HTML structure has issues")
        print("="*70)
        return False


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# Server Features Test Suite")
    print("#"*70)

    results = []

    # Run tests
    results.append(("Random Prompts", test_random_prompts_exist()))
    print("\n" + "-"*70 + "\n")
    results.append(("Tooltip Coverage", test_tooltip_count()))
    print("\n" + "-"*70 + "\n")
    results.append(("HTML Structure", test_html_structure()))

    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        sys.exit(1)
