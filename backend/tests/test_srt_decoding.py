import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.srt_parser import _TIMING_RE, parse_srt

def test_srt_decoding():
    srt_content = """1
00:00:00,896 --> 00:00:08,999
Subviet: yugo9x

2
00:00:09,242 --> 00:00:13,303
Anh mất sự cân bằng và rối trí
"""
    # Test UTF-8 bytes
    utf8_bytes = srt_content.encode("utf-8")
    
    # Test UTF-16LE bytes
    utf16_bytes = srt_content.encode("utf-16")
    
    # Try our decoding logic
    for raw_bytes in [utf8_bytes, utf16_bytes]:
        text = None
        for enc in ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"]:
            try:
                t = raw_bytes.decode(enc)
                if _TIMING_RE.search(t):
                    text = t
                    break
            except UnicodeDecodeError:
                continue
        assert text is not None
        result = parse_srt(text)
        assert len(result.segments) == 2
        assert result.segments[0]["text"] == "subviet: yugo9x"
        assert result.segments[0]["text_original"] == "Subviet: yugo9x"
        assert result.segments[1]["text"] == "anh mất sự cân bằng và rối trí"
        assert result.segments[1]["text_original"] == "Anh mất sự cân bằng và rối trí"
        
    print("SRT decoding tests passed successfully!")


def test_srt_lowercases_all_caps_cues():
    srt_content = """1
00:00:01,000 --> 00:00:04,000
XIN CHÀO MỌI NGƯỜI!

2
00:00:05,000 --> 00:00:08,000
THIS IS A TEST
"""
    result = parse_srt(srt_content)
    assert len(result.segments) == 2
    assert result.segments[0]["text"] == "xin chào mọi người!"
    assert result.segments[0]["text_original"] == "XIN CHÀO MỌI NGƯỜI!"
    assert result.segments[1]["text"] == "this is a test"
    assert result.segments[1]["text_original"] == "THIS IS A TEST"
    print("SRT lowercase normalization tests passed successfully!")

if __name__ == "__main__":
    test_srt_decoding()
