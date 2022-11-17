%% Read Pivot File
% Description: Generic function for reading text format data files.
%
% Output: output | Data Matrices | 1xN Cell Array
%
% Input:    path | File Path     | String
%
% Created by: Ethan Elgavish | 04-23-2022
% Edited by: Ethan Elgavish | 04-23-2022
function output = readPivotFile(path)
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
    headerLength = length(header);
    
    Ninframe = [];
    for i = 1 : headerLength-2
        Ninframe(i) = str2num(header{i});
    end
    
    j = 1;
    Nframes = str2num(header{headerLength-1});
    for i = 1:Nframes
        for k = 1:length(Ninframe)
            inframe = Ninframe(k);
            output{length(output)+1} = M(j:j+inframe-1, 1:n);
            j = j+inframe;
        end
    end
    
    fclose(fileID);
end