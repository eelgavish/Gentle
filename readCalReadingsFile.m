%% Read Calibration Readings File
% Description: Generic function for reading text format data files.
%
% Output: output | Data Matrices | 1xN Cell Array
%
% Input:    path | File Path     | String
%
% Created by: Ethan Elgavish | 04-18-2022
% Edited by: Ethan Elgavish | 04-25-2022
function output = readCalReadingsFile(path)
    output = {};

    [fileID,errmsg] = fopen(path);
    if fileID < 0
        disp(errmsg);
        return;
    end
    
    M = csvread(path,1,0);
    [m, n] = size(M);
    
    header = fgetl(fileID);
    header = strsplit(header, ',');
    L = length(header);
    numFrames = str2num(header{L-1});
    
    j = 1;
    for frame=1:numFrames
        for i = 1 : L-2
            k = j+str2num(header{i})-1;
            output{(frame-1)*(L-2)+i} = M(j:k, 1:n);
            j = k+1;
        end
    end
    
    fclose(fileID);
end