# FIX Symbol Discrepancy Checker

## Overview
This tool identifies discrepancies between FIX message symbols and the security master database, focusing on detecting corporate action-related symbol changes that weren't properly updated in trading systems.

## Quick Start
```bash
# Basic usage with default paths
python -m fix_checker.main

# Custom path specification
python -m fix_checker.main --secmaster /path/to/secmaster.csv --fix-log /path/to/fixlog.txt --output-file /path/to/report.csv
```

## Input File Formats

### Security Master CSV
Required format:
```
CUSIP,Symbol,Name,Exchange
037833100,AAPL,"APPLE INC",NASDAQ
594918104,MSFT,"MICROSOFT CORP",NASDAQ
...
```

### FIX Log Format
The tool processes standard FIX 4.2 messages using SOH (|) as delimiter:
```
8=FIX.4.2|9=211|35=D|34=8|...|48=037833100|54=1|55=AAPL|...
8=FIX.4.2|9=219|35=8|34=9|...|48=037833100|54=1|55=AAPL|...
```

Only processes message types:
- NewOrderSingle (35=D)  
- ExecutionReport (35=8)

## Output Report
The discrepancy report is a CSV file sorted by financial exposure:

```
CUSIP,MasterSymbol,FIXSymbol,Quantity,Price,Exposure,Exchange
037833100,AAPL,APL,1000,142.15,142150.00,NASDAQ
...
```

## Command-Line Options
```
--secmaster PATH      Path to security master CSV file
--fix-log PATH        Path to FIX log file
--output-file PATH    Path to output report
--log-level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
--delimiter CHAR      FIX message delimiter (default '|')
```

## Exit Codes
- 0: Success
- 1: Configuration error
- 2: File access error
- 3: Processing error

## Troubleshooting
- Verify file permissions and paths
- Check CSV header format matches expected fields
- Ensure FIX messages contain required fields (48=CUSIP, 55=Symbol)
- For debugging, use `--log-level DEBUG`
