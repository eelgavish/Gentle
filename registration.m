%% Registration
% Description: Generic registration function for finding transorm between
%              two point sets with known correspondence.
%
% Output:   T | Transform: S = T*M | 
%
% Input:  M | Moving set of points | Nx3 Matrix
%         S | Static set of points | Nx3 Matrix
%
% Created by: Ethan Elgavish | 04-18-2022
% Edited by: Ethan Elgavish | 04-18-2022
function T = registration(M, S)
    [m1,n1] = size(M);
    [m2,n2] = size(S);
    if m1 ~= m2
        disp('-FATAL- registration: Input sets do not have same number of points, aborting. May need to use an unknown correspondence registration method.');
        return;
    end
    if n1 ~= 3
        disp('-FATAL- registration: Input set 1 does not have 3 elements per row, aborting.');
        return;
    elseif n2 ~= 3
        disp('-FATAL- registration: Input set 2 does not have 3 elements per row, aborting.');
        return;
    end
    
    % Find centers of mass
    longone = ones([1,m1]);
    Mcm = longone*M/m1;
    Scm = longone*S/m2;
    
    t = (Scm - Mcm).';
    
    % Find rotation using SVD
    H = zeros([3, 3]);
    for i=1:m1
        mi = M(i,1:3)-Mcm;
        si = S(i,1:3)-Scm;
        Hi = [[mi(1)*si(1) mi(1)*si(2) mi(1)*si(3)];
              [mi(2)*si(1) mi(2)*si(2) mi(2)*si(3)];
              [mi(3)*si(1) mi(3)*si(2) mi(3)*si(3)]];
        H = H + Hi;
    end
    [U,~,V] = svd(H); 
    R = V * U.';
    
    if det(R)-1 > 1e-3
        disp('-ERROR- registration: Determinant of R is not equal to 1, solution may be invalid.');
    end
    
    T = [[R t];
         [0 0 0 1]];
end