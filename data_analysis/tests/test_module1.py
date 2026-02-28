"""
Test Module 1 using the exact example log data provided.
Constructs a synthetic .txt file from the four example blocks,
runs the parser, and checks the outputs.
"""

import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.module1_behavioral import process_subject

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)

# ---------------------------------------------------------------------------
# Synthetic E-Prime .txt built from the exact examples provided
# ---------------------------------------------------------------------------
# Note: row counts are deliberately small (2 encoding, 2 distractor, 2 decoding,
# 2 exploratory) so the validation warnings for wrong counts are expected here.

SAMPLE_TXT = """\
*** LogFrame Start ***
		Procedure: PracticeProc
		PracticeList: 14
		CueText: blender
		StimID: 2349475
		Question: Where is the man (powering the blender) positioned relative to the blender?
		Opt1: to the left of it
		Opt2: behind it
		Opt3: in front of it
		Opt4: to the right of it
		CorrectKey: 1
		ImageFile: .\\stimuli\\2349475.jpg
		Running: PracticeList
		PracticeList.Cycle: 1
		PracticeList.Sample: 1
		Practice4AFC.DEVICE: Keyboard
		Practice4AFC.OnsetAckDelay: 0
		Practice4AFC.OnsetDelay: 816
		Practice4AFC.OnsetTime: 197806
		Practice4AFC.DurationError: -999999
		Practice4AFC.RTTime: 217167
		Practice4AFC.ACC: 1
		Practice4AFC.RT: 19361
		Practice4AFC.RESP: 1
		Practice4AFC.CRESP: 1
		Practice4AFC.OnsetToOnsetTime: 0
		Practice4AFC.Choice.Value: 
		YourAnswerText: to the left of it
		CorrectAnswerText: to the left of it
*** LogFrame End ***
*** LogFrame Start ***
		Procedure: PracticeProc
		PracticeList: 5
		CueText: football
		StimID: 2345035
		Question: Who runs toward the football in the scene?
		Opt1: girl and boy
		Opt2: girls
		Opt3: boys
		Opt4: men
		CorrectKey: 2
		ImageFile: .\\stimuli\\2345035.jpg
		Running: PracticeList
		PracticeList.Cycle: 1
		PracticeList.Sample: 2
		Practice4AFC.ACC: 0
		Practice4AFC.RT: 6578
		Practice4AFC.RESP: 1
		Practice4AFC.CRESP: 2
		YourAnswerText: girl and boy
		CorrectAnswerText: girls
*** LogFrame End ***
*** LogFrame Start ***
		Procedure: DistractorProc
		StimID: 1051
		Question: 73 - 29 = ?
		Opt1: 54
		Opt2: 41
		Opt3: 44
		Opt4: 48
		CorrectKey: 3
		DistractorList: 51
		Running: DistractorList
		DistractorList.Cycle: 1
		DistractorList.Sample: 4
		Distractor4AFC.ACC: 0
		Distractor4AFC.RT: 4614
		Distractor4AFC.RESP: 2
		Distractor4AFC.CRESP: 3
		Distractor4AFC.DEVICE: Keyboard
*** LogFrame End ***
*** LogFrame Start ***
		Procedure: DistractorProc
		StimID: 1145
		Question: 12 * 4 = ?
		Opt1: 48
		Opt2: 43
		Opt3: 44
		Opt4: 54
		CorrectKey: 1
		DistractorList: 145
		Running: DistractorList
		DistractorList.Cycle: 1
		DistractorList.Sample: 5
		Distractor4AFC.ACC: 1
		Distractor4AFC.RT: 3916
		Distractor4AFC.RESP: 1
		Distractor4AFC.CRESP: 1
		Distractor4AFC.DEVICE: Keyboard
*** LogFrame End ***
*** LogFrame Start ***
		Procedure: TestProc
		CueText: horse ride
		StimID: 2394020
		Question: Which description best matches the image?
		Opt1: horses ride side-by-side on beach
		Opt2: horses follow one another through river
		Opt3: horses face each other on beach
		Opt4: horses walk away from the beach
		CorrectKey: 1
		ImageFile: .\\stimuli\\2394020.jpg
		TestList: 27
		Running: TestList
		TestList.Cycle: 1
		TestList.Sample: 19
		Test4AFC.ACC: 0
		Test4AFC.RT: 8839
		Test4AFC.RESP: 3
		Test4AFC.CRESP: 1
		RateConf.ACC: 0
		RateConf.RT: 1508
		RateConf.RESP: 4
		RateConf.CRESP: 1
*** LogFrame End ***
*** LogFrame Start ***
		Procedure: TestProc
		CueText: wine
		StimID: 2347400
		Question: Which description best matches the image?
		Opt1: men clink wine glasses
		Opt2: men grab wine bottles
		Opt3: men sip from wine glasses
		Opt4: men smell wine glasses
		CorrectKey: 4
		ImageFile: .\\stimuli\\2347400.jpg
		TestList: 5
		Running: TestList
		TestList.Cycle: 1
		TestList.Sample: 20
		Test4AFC.ACC: 0
		Test4AFC.RT: 9824
		Test4AFC.RESP: 1
		Test4AFC.CRESP: 4
		RateConf.RT: 3026
		RateConf.RESP: 3
		RateConf.CRESP: 4
*** LogFrame End ***
*** LogFrame Start ***
		Procedure: ExploratoryProc
		CueText: laptop
		StimID: 2410309
		Question: Which description best matches the image?
		Opt1: man types on laptop keyboard
		Opt2: woman and baby use laptops
		Opt3: baby touches laptop mouse
		Opt4: woman closes laptop lid
		CorrectKey: 2
		ImageFile: .\\stimuli\\2410309.jpg
		ExploratoryList: 30
		Running: ExploratoryList
		ExploratoryList.Cycle: 2
		ExploratoryList.Sample: 7
		FreeResponseSlide.RT: 31833
		FreeResponseSlide.RESP: babywascopyingtmumonthesofaswhoithinkhadadietcokitlookedlikesheneededihebabywastyp
		FreeResponseSlide.CRESP: 
*** LogFrame End ***
*** LogFrame Start ***
		Procedure: ExploratoryProc
		CueText: umbrella
		StimID: 2347025
		Question: Which description best matches the image?
		Opt1: baby lies under umbrella
		Opt2: woman holds umbrella over baby
		Opt3: baby holds onto umbrella
		Opt4: baby drags pink umbrella
		CorrectKey: 3
		ImageFile: .\\stimuli\\2347025.jpg
		ExploratoryList: 4
		Running: ExploratoryList
		ExploratoryList.Cycle: 2
		ExploratoryList.Sample: 8
		FreeResponseSlide.RT: 70342
		FreeResponseSlide.RESP: babyonthemumsbackholdingnumberellitisweighinionthemumsshouldermaybebcidkifthekidisstingenoughtoholdthathemumhadafunnyturbanhtthinIwouldguesstheywereasianmaybebcofislamichabutnotsoutheastasialookedmountainypeopl
		FreeResponseSlide.CRESP: 
*** LogFrame End ***
"""


def run_tests():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "data_behavioral"
        output_dir = tmpdir / "output" / "behavioral"
        input_dir.mkdir()

        # Write synthetic file
        (input_dir / "sub_test.txt").write_text(SAMPLE_TXT, encoding="utf-8")

        # Run parser
        ok = process_subject("sub_test", input_dir, output_dir)
        assert ok, "process_subject returned False"

        # Load outputs — force StimID as string since pandas would otherwise
        # infer it as int64 (the values happen to look numeric)
        read_kwargs = {"dtype": {"StimID": str}}
        enc = pd.read_csv(output_dir / "sub_test_encoding.csv", **read_kwargs)
        dist = pd.read_csv(output_dir / "sub_test_distractor.csv", **read_kwargs)
        dec = pd.read_csv(output_dir / "sub_test_decoding.csv", **read_kwargs)
        exp = pd.read_csv(output_dir / "sub_test_exploratory.csv", **read_kwargs)

        print("\n=== ENCODING ===")
        print(enc.to_string())

        print("\n=== DISTRACTOR ===")
        print(dist.to_string())

        print("\n=== DECODING ===")
        print(dec.to_string())

        print("\n=== EXPLORATORY ===")
        print(exp.to_string())

        # --- Assertions ---

        # SubjectID present everywhere
        for df, name in [
            (enc, "encoding"),
            (dist, "distractor"),
            (dec, "decoding"),
            (exp, "exploratory"),
        ]:
            assert "SubjectID" in df.columns, f"SubjectID missing from {name}"
            assert (df["SubjectID"] == "sub_test").all(), f"SubjectID wrong in {name}"

        # Encoding: 2 rows, correct values
        assert len(enc) == 2
        assert enc.loc[0, "StimID"] == "2349475"
        assert enc.loc[0, "CueText"] == "blender"
        assert enc.loc[0, "Accuracy"] == 1
        assert enc.loc[0, "RT_ms"] == 19361
        assert enc.loc[1, "Accuracy"] == 0

        # Distractor: 2 rows, no CueText column
        assert len(dist) == 2
        assert "CueText" not in dist.columns
        assert dist.loc[0, "StimID"] == "1051"
        assert dist.loc[0, "Accuracy"] == 0

        # Decoding: 2 rows, confidence present
        assert len(dec) == 2
        assert dec.loc[0, "StimID"] == "2394020"
        assert dec.loc[0, "CueText"] == "horse ride"
        assert dec.loc[0, "Accuracy"] == 0
        assert dec.loc[0, "Confidence"] == 4
        assert dec.loc[1, "Confidence"] == 3

        # Exploratory: 2 rows, FreeResponse present and non-empty
        assert len(exp) == 2
        assert exp.loc[0, "StimID"] == "2410309"
        assert exp.loc[0, "CueText"] == "laptop"
        assert isinstance(exp.loc[0, "FreeResponse"], str)
        assert len(exp.loc[0, "FreeResponse"]) > 10
        assert exp.loc[1, "CueText"] == "umbrella"

        # StimID is string type (could be object or StringDtype depending on pandas version)
        assert enc["StimID"].apply(lambda x: isinstance(x, str)).all()
        assert dec["StimID"].apply(lambda x: isinstance(x, str)).all()

        print("\n✓ All assertions passed.")


if __name__ == "__main__":
    run_tests()
