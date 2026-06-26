import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.segmentation import assign_speakers_from_turns, assign_speakers_from_diarization

# Mock pyannote annotation structure
class MockTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class MockDiarization:
    def __init__(self, tracks):
        self.tracks = tracks # list of (turn, None, speaker)

    def itertracks(self, yield_label=True):
        return self.tracks

def test_assign_speakers_from_turns_no_split():
    segments = [
        {
            "id": "s1",
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                {"start": 0.0, "end": 0.5, "text": "hello"},
                {"start": 0.5, "end": 1.0, "text": "world"}
            ]
        }
    ]
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 1"}
    ]
    res = assign_speakers_from_turns(segments, turns)
    assert len(res) == 1
    assert res[0]["speaker_id"] == "Speaker 1"
    assert res[0]["id"] == "s1"
    assert res[0]["text"] == "hello world"
    print("test_assign_speakers_from_turns_no_split passed")

def test_assign_speakers_from_turns_with_split():
    segments = [
        {
            "id": "s1",
            "start": 0.0,
            "end": 3.0,
            "text": "hello world how are you",
            "words": [
                {"start": 0.0, "end": 0.5, "text": "hello"},
                {"start": 0.5, "end": 1.0, "text": "world"},
                {"start": 1.5, "end": 2.0, "text": "how"},
                {"start": 2.0, "end": 2.5, "text": "are"},
                {"start": 2.5, "end": 3.0, "text": "you"}
            ]
        }
    ]
    turns = [
        {"start": 0.0, "end": 1.2, "speaker": "Speaker 1"},
        {"start": 1.3, "end": 3.0, "speaker": "Speaker 2"}
    ]
    res = assign_speakers_from_turns(segments, turns)
    assert len(res) == 2
    assert res[0]["id"] == "s1_0"
    assert res[0]["speaker_id"] == "Speaker 1"
    assert res[0]["text"] == "hello world"
    assert res[0]["start"] == 0.0
    assert res[0]["end"] == 1.0

    assert res[1]["id"] == "s1_1"
    assert res[1]["speaker_id"] == "Speaker 2"
    assert res[1]["text"] == "how are you"
    assert res[1]["start"] == 1.5
    assert res[1]["end"] == 3.0
    print("test_assign_speakers_from_turns_with_split passed")

def test_assign_speakers_from_diarization_with_split():
    segments = [
        {
            "id": "s2",
            "start": 10.0,
            "end": 13.0,
            "text": "what is this split test",
            "words": [
                {"start": 10.0, "end": 10.5, "text": "what"},
                {"start": 10.5, "end": 11.0, "text": "is"},
                {"start": 11.5, "end": 12.0, "text": "this"},
                {"start": 12.0, "end": 12.5, "text": "split"},
                {"start": 12.5, "end": 13.0, "text": "test"}
            ]
        }
    ]
    diarization = MockDiarization([
        (MockTurn(9.0, 11.2), None, "SPEAKER_00"),
        (MockTurn(11.3, 14.0), None, "SPEAKER_01"),
    ])
    res = assign_speakers_from_diarization(segments, diarization)
    assert len(res) == 2
    assert res[0]["id"] == "s2_0"
    assert res[0]["speaker_id"] == "Speaker 1"
    assert res[0]["text"] == "what is"
    
    assert res[1]["id"] == "s2_1"
    assert res[1]["speaker_id"] == "Speaker 2"
    assert res[1]["text"] == "this split test"
    print("test_assign_speakers_from_diarization_with_split passed")

def test_assign_speakers_from_turns_sentence_boundary_split():
    segments = [
        {
            "id": "s3",
            "start": 0.0,
            "end": 3.0,
            "text": "hello. world how are you",
            "words": [
                {"start": 0.0, "end": 0.5, "text": "hello."},
                {"start": 0.5, "end": 1.0, "text": "world"},
                {"start": 1.5, "end": 2.0, "text": "how"},
                {"start": 2.0, "end": 2.5, "text": "are"},
                {"start": 2.5, "end": 3.0, "text": "you"}
            ]
        }
    ]
    # Even if turns assigns same speaker to everything, it should split at the period sentence boundary!
    turns = [
        {"start": 0.0, "end": 3.0, "speaker": "Speaker 1"}
    ]
    res = assign_speakers_from_turns(segments, turns)
    assert len(res) == 2
    assert res[0]["id"] == "s3_0"
    assert res[0]["speaker_id"] == "Speaker 1"
    assert res[0]["text"] == "hello."
    assert res[0]["start"] == 0.0
    assert res[0]["end"] == 0.5

    assert res[1]["id"] == "s3_1"
    assert res[1]["speaker_id"] == "Speaker 1"
    assert res[1]["text"] == "world how are you"
    assert res[1]["start"] == 0.5
    assert res[1]["end"] == 3.0
    print("test_assign_speakers_from_turns_sentence_boundary_split passed")

if __name__ == "__main__":
    test_assign_speakers_from_turns_no_split()
    test_assign_speakers_from_turns_with_split()
    test_assign_speakers_from_diarization_with_split()
    test_assign_speakers_from_turns_sentence_boundary_split()
    print("All tests passed successfully!")
