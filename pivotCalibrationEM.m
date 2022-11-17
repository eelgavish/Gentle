%% EM Tracker Pivot Calibration
% Description: Pivot calibration method for EM tracker.
%
% Output:  btip | Tip length    | Vector
%         bpost | Post position | Vector
%
% Input:   M | Full set of point sets | 1xN Cell Array
%
% Created by: Ethan Elgavish | 04-23-2022
% Edited by: Ethan Elgavish | 04-23-2022
function [btip, bpost] = pivotCalibrationEM(M)
    numSets = length(M);
    Rk = zeros([numSets*3, 6]);
    Pk = zeros([numSets*3, 1]);
    
    for i=1:numSets
        
        % Grab first set
        set = M{i};
        
        [m1, n1] = size(set);
        if n1 ~= 3
            disp("-FATAL- registration: Input set " + i + " does not have 3 elements per row, aborting.");
            return;
        end
        
        % Find geometric center
        longone = ones([1,m1]);
        Pk((i-1)*3+1:(i-1)*3+3) = -longone*set/m1;
        
        if i == 1
            Rk(1:3, 1:6) = [eye(3), -eye(3)];
        else
            T = registration(M{1}, set);
            Rk((i-1)*3+1:(i-1)*3+3, 1:6) = [T(1:3,1:3), -eye(3)];
        end
    end
    
    b = Rk\Pk;
    
    btip = b(1:3);
    bpost = b(4:6);
end