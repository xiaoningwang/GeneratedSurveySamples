from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'survey_data'
OUT=ROOT/'paper_results'
DETAILS=OUT/'details'
MECHANISM=DETAILS/'mechanism'
API=DETAILS/'api'
R=2000
K=3

def create_folders():
    for path in [OUT,DETAILS,MECHANISM,API]:path.mkdir(parents=True,exist_ok=True)
