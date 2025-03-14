# shaw-fixrec-demo
# Post Interview Python Demo Project: FIX Reconciliation Script

Version 1:
Start TIME: 14:30 (Friday)
End Time:

## Overview
This tool identifies discrepancies between FIX message symbols and the security master database, with focus on detecting corporate action-related symbol changes that weren't properly updated in trading systems.

## Features
- Parses FIX 4.2 protocol messages (NewOrderSingle and ExecReport)
- Identifies symbol mismatches where CUSIP matches
- Calculates financial exposure (Price × Quantity) for each discrepancy
- Ranks discrepancies by financial impact for prioritization
- Generates CSV reports for operations teams

## Implementation Notes
This project was implemented with a 12-hour turnaround goal to demonstrate rapid problem-solving skills. Some compromises were made to prioritize core functionality:

- Testing is minimal (would add pytest coverage in production)
- Error handling focuses on critical paths only
- Logging is simplified
- Documentation is condensed

In a production environment, I would address these limitations and add comprehensive test coverage.

## Usage
```
python main.py --secmaster data/security_master.csv --fix-log data/fix_messages.txt --output reports/discrepancy_report.csv
```

## Requirements
- Python 3.8+
- pandas

## Files
- `config.py` - Configuration and CLI arguments
- `models.py` - Data structures
- `security_master.py` - Reference data loading
- `fix_parser.py` - FIX protocol parsing
- `analyzer.py` - Discrepancy detection logic
- `reporter.py` - Report generation
- `main.py` - Program entry point
