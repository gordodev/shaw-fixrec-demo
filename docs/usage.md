# FIX Symbol Discrepancy Checker

## Overview
This tool identifies discrepancies between FIX message symbols and the security master database, focusing on detecting corporate action-related symbol changes that weren't properly updated in trading systems.

## Quick Start
```bash
# Basic usage with default paths
python main.py

# Custom path specification
python main.py \
  --secmaster /path/to/secmaster.csv \
  --fix-log /path/to/fixlog.txt \
  --output /path/to/report.csv \
  --log-level DEBUG
```

## Input File Formats

### Security Master CSV
Required columns:
```
CUSIP,Symbol,Description,Region,Exchange,Asset Class,Currency,Country
037833100,AAPL,Apple Inc. Common Stock,US,NASDAQ,Equity,USD,USA
594918104,MSFT,Microsoft Corp. Common Stock,US,NASDAQ,Equity,USD,USA
```

### FIX Log Format
Processes FIX 4.2 messages using delimiter (default '|'):
```
8=FIX.4.2|9=211|35=D|34=8|...|48=037833100|54=1|55=AAPL|38=1000|44=142.15|...
8=FIX.4.2|9=219|35=8|34=9|...|48=037833100|54=1|55=AAPL|...
```

Supported message types:
- NewOrderSingle (35=D)  
- ExecutionReport (35=8)

## Output Report
CSV file sorted by financial exposure:
```
CUSIP,MasterSymbol,FIXSymbol,Quantity,Price,Exposure,Exchange,Region
037833100,AAPL,APL,1000,142.15,142150.00,NASDAQ,US
...
```

## Command-Line Options
```
--secmaster PATH      Path to security master CSV file
--fix-log PATH        Path to FIX log file
--output PATH         Path to output report
--log-level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
--delimiter CHAR      FIX message delimiter (default '|')
--chunk-size MB       Security master chunk size in MB (default: 64)
--disable-mmap        Disable memory mapping for file loading
```

## Performance Features
- Memory-efficient chunked processing
- Supports large files (tested up to 200GB)
- Real-time progress monitoring
- Resource usage tracking

## Exit Codes
- 0: Success
- 1: General error
- 2: File access error
- 3: Data validation error

## Troubleshooting
- Verify file permissions and paths
- Check CSV header format
- Ensure FIX messages contain required fields
- Use `--log-level DEBUG` for detailed diagnostics
- Monitor console output for processing details
