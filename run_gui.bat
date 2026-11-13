@echo off
chcp 65001 > nul
echo =====================================================================
echo    KHOI DONG HE THONG MO PHONG & TAI TAO CT (CT SIMULATOR PRO)
echo    Dai hoc Quoc gia TP.HCM - Truong Dai hoc Bach khoa
echo    Khoa Khoa hoc Ung dung - Bo mon Vat ly Y sinh
echo =====================================================================
echo Dang khoi chay giao dien do hoa GUI...

if exist "dist\CT_Simulator.exe" (
    start "" "dist\CT_Simulator.exe"
) else (
    python main.py
)
