@echo off
echo ================================================================================
echo MARKETLAB 2.0 - BASELINE PIPELINE
echo PC 2: Processing stocks 18-34
echo ================================================================================
echo.
echo Starting baseline pipeline...
echo This will take approximately 3-5 hours.
echo.
echo You can:
echo - Close this window anytime (progress is saved)
echo - Rerun this file to resume from checkpoint
echo.
pause

python baseline_pipeline.py 2

echo.
echo ================================================================================
echo COMPLETE!
echo ================================================================================
pause