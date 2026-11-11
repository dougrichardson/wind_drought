import pandas as pd

def read_swt():
    # Australian Synoptic Weather Type data from forked repository
    return pd.read_csv(
        "../Australian-synoptic-weather-types/SWT_climatology/SWT_climatology.csv",
        index_col=0,
        parse_dates=True
    )

def get_ordered_swt():
    return [
        "WH-A", "WH-B", "WH-C", "WH-D",
        "CH-A", "CH-B",
        "EH-A", "EH-B", "EH-C", "EH-D", "EH-E",
        "TH-A", "TH-B", "TH-C",
        "FH-A", "FH-B", "FH-C",
        "WCT-A", "WCT-B",
        "COL-A", "COL-B", "COL-C", "COL-D", "COL-E", "COL-F",
        "AM-A", "AM-B", "AM-C", "AM-D", "AM-E"
    ]

def get_swt_colors():
    """Return dict of color for each SWT"""
    swt_colors = [
        (134/255,0/255,34/255),(241/255,0/255,241/255),(255/255,134/255,255/255),(255/255,241/255,255/255), #Pink WH
        (255/255,204/255,51/255),(255/255,245/255,204/255), #Yellow CH
        (153/255,15/255,15/255),(178/255,44/255,44/255),(204/255,81/255,81/255),(229/255,126/255,126/255),(255/255,178/255,178/255), #Red EH
        (153/255,84/255,15/255),(204/255,142/255,81/255),(255/255,216/255, 178/255), #Brown TH
        (107/255,153/255,15/255),(163/255,204/255,81/255),(195/255,229/255,126/255), #Green FH (133/255,178/255,44/255),
        (66/255,44/255,178/255),(143/255,126/255,229/255), #Purple WCT
        (5/255,67/255,113/255),(15/255,107/255,153/255),(44/255,133/255,178/255),(81/255,163/255,204/255),(126/255,195/255,229/255),(178/255,229/255,255/255), #Blue COL
        (0/255,60/255,48/255),(1/255,102/255,95/255),(53/255,151/255,143/255),(128/255,205/255,193/255),(199/255,234/255,229/255)
    ]
    return dict(zip(get_ordered_swt(), swt_colors))