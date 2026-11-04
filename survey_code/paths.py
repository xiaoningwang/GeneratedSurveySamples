from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'survey_data'
# A separate output directory lets reviewers retain the published reference files.
OUT=Path(os.environ.get('SURVEY_OUTPUT_DIR', str(ROOT/'paper_results'))).resolve()
DETAILS=OUT/'details'
MECHANISM=DETAILS/'mechanism'
API=DETAILS/'api'
R=2000
K=3

def create_folders():
    for path in [OUT,DETAILS,MECHANISM,API]:path.mkdir(parents=True,exist_ok=True)
