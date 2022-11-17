M = readPivotFile('IMUdata/Calibration_DrVanardosanastomosis1667251523.csv');
[btip, bpost] = pivotCalibrationEM(M);
btip
bpost