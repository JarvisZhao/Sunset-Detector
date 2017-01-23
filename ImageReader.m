function [Means, STDs, L, S, T] = ImageReader(directory)
fileList = dir(directory);

for i = 3:size(fileList)
    j=size(fileList,1)+3-i
    [directory  '/'  fileList(j).name]
    
    img = imread([directory  '/'  fileList(j).name]);
    
    [Mean Deviation L S T] = extractFeature(img);
    Means(:,:,i) = Mean;
    STDs(:,:,i) = Deviation;
end


function [Mean, Deviation, L, S, T] = extractFeature(image)

R = double(image(:,:,1));
G = double(image(:,:,2));
B = double(image(:,:,3));

L = double(R + G + B);
S = double(R - B);
T = double(R -2*G + B);
tempImg(:,:,1) =L;
tempImg(:,:,2) =S;
tempImg(:,:,3) =T;
a = size(image,1);
b = size(image,2);

Ascale = uint8(a/7);
Bscale = uint8(b/7);

for i = 1:7
    firstA = (i-1)*Ascale+1;
    secondA = i*Ascale;
    
    for j = 1:7
        firstB = (i-1)*Bscale+1;
        secondB = i*Bscale;
       
        meanL(7*(i-1)+j) = mean(mean(L(firstA:secondA, firstB:secondB)));
        meanS(7*(i-1)+j) = mean(mean(S(firstA:secondA, firstB:secondB)));
        meanT(7*(i-1)+j) = mean(mean(T(firstA:secondA, firstB:secondB)));
        stdL(7*(i-1)+j) = std2(L(firstA:secondA, firstB:secondB));
        stdS(7*(i-1)+j) = std2(S(firstA:secondA, firstB:secondB));
        stdT(7*(i-1)+j) = std2(T(firstA:secondA, firstB:secondB));
        
    end
end

Mean(:,1)=meanL;
Mean(:,2)=meanS;
Mean(:,3)=meanT;

Deviation(:,1) = stdL;
Deviation(:,2) = stdS;
Deviation(:,3) = stdT;
