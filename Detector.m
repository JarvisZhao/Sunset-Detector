%%

img = imread('sunsetDetectorImagesWithDifficult\sunsetDetectorImages\TrainSunset\101.jpg');
a = uint16(size(img,1));
b = uint16(size(img,2));
for i = 1:6
   img(a/7*i,:,1)=255;
   img(a/7*i,:,2)=255;
   img(a/7*i,:,3)=255;
   
end

for i = 1:6
      img(:,b/7*i,1)=255;
      img(:,b/7*i,2)=255;
      img(:,b/7*i,3)=255; 
end
imtool(img);
%%
%[PosTrainMeans PosTrainSTDs PosTrainL PosTrainS PosTrainT] = ImageReader('sunsetDetectorImagesWithDifficult\sunsetDetectorImages\TrainSunset');

%[NegTrainMeans NegTrainSTDs NegTrainL NegTrainS NegTrainT] = ImageReader('sunsetDetectorImagesWithDifficult\sunsetDetectorImages\TrainNonsunsets');
[DPosTestMeans DPosTestSTDs DPosTestL DPosTestS DPosTestT] = ImageReader('sunsetDetectorImagesWithDifficult\sunsetDetectorImages\TestDifficultSunsets');
[NegTestMeans NegTestSTDs NegTestL NegTestS NegTestT] = ImageReader('sunsetDetectorImagesWithDifficult\sunsetDetectorImages\TestNonsunsets');
[DNegTestMeans DNegTestSTDs DNegTestL  DNegTestT] = ImageReader('sunsetDetectorImagesWithDifficult\sunsetDetectorImages\TestDifficultNonsunsets');
[PosTestMeans PosTestSTDs PosTestL PosTestS PosTestT] = ImageReader('sunsetDetectorImagesWithDifficult\sunsetDetectorImages\TestSunset');


%%
for i = 1:49
    
    
    NegTestFeaturesAll(:,1+(i-1)*6) = NegTestMeans(i,1,:);
    NegTestFeaturesAll(:,3+(i-1)*6) = NegTestMeans(i,2,:);
    NegTestFeaturesAll(:,5+(i-1)*6) = NegTestMeans(i,3,:);
    NegTestFeaturesAll(:,2+(i-1)*6) = NegTestSTDs(i,1,:);
    NegTestFeaturesAll(:,4+(i-1)*6) = NegTestSTDs(i,2,:);
    NegTestFeaturesAll(:,6+(i-1)*6) = NegTestSTDs(i,3,:);
    
    DNegTestFeaturesAll(:,1+(i-1)*6) = DNegTestMeans(i,1,:);
    DNegTestFeaturesAll(:,3+(i-1)*6) = DNegTestMeans(i,2,:);
    DNegTestFeaturesAll(:,5+(i-1)*6) = DNegTestMeans(i,3,:);
    DNegTestFeaturesAll(:,2+(i-1)*6) = DNegTestSTDs(i,1,:);
    DNegTestFeaturesAll(:,4+(i-1)*6) = DNegTestSTDs(i,2,:);
    DNegTestFeaturesAll(:,6+(i-1)*6) = DNegTestSTDs(i,3,:);
    
    PosTestFeaturesAll(:,1+(i-1)*6) = PosTestMeans(i,1,:);
    PosTestFeaturesAll(:,3+(i-1)*6) = PosTestMeans(i,2,:);
    PosTestFeaturesAll(:,5+(i-1)*6) = PosTestMeans(i,3,:);
    PosTestFeaturesAll(:,2+(i-1)*6) = PosTestSTDs(i,1,:);
    PosTestFeaturesAll(:,4+(i-1)*6) = PosTestSTDs(i,2,:);
    PosTestFeaturesAll(:,6+(i-1)*6) = PosTestSTDs(i,3,:);
    
    DPosTestFeaturesAll(:,1+(i-1)*6) = DPosTestMeans(i,1,:);
    DPosTestFeaturesAll(:,3+(i-1)*6) = DPosTestMeans(i,2,:);
    DPosTestFeaturesAll(:,5+(i-1)*6) = DPosTestMeans(i,3,:);
    DPosTestFeaturesAll(:,2+(i-1)*6) = DPosTestSTDs(i,1,:);
    DPosTestFeaturesAll(:,4+(i-1)*6) = DPosTestSTDs(i,2,:);
    DPosTestFeaturesAll(:,6+(i-1)*6) = DPosTestSTDs(i,3,:);
    
end


%%
load('testFeatures.mat');
load('trainFeatures.mat');
PosTestPredict = ones(size(PosTestFeaturesAll,1),1);
NegTestPredict = -1*ones(size(NegTestFeaturesAll,1),1);
DPosTestPredict = ones(size(DPosTestFeaturesAll,1),1);
DNegTestPredict = -1*ones(size(DNegTestFeaturesAll,1),1);
PosTestPredict = ones(size(PosTestFeaturesAll,1),1);
allF = vertcat(NegTestFeaturesAll,DNegTestFeaturesAll,PosTestFeaturesAll,DPosTestFeaturesAll);
allF = vertcat(allF, PosFeaturesAll, NegFeaturesAll);

[mins maxs normalized] = normalizeFeatures01(allF);
%%
for i = 1:6
   for j =1:49
   NegTestFeaturesAll(:,(j-1)*6+i) =  (NegTestFeaturesAll(:,(j-1)*6+i)-mins(i))/maxs(i);
   DNegTestFeaturesAll(:,(j-1)*6+i) = (DNegTestFeaturesAll(:,(j-1)*6+i)-mins(i))/maxs(i);
   PosTestFeaturesAll(:,(j-1)*6+i) = (PosTestFeaturesAll(:,(j-1)*6+i)-mins(i))/maxs(i);
   DPosTestFeaturesAll(:,(j-1)*6+i) = (DPosTestFeaturesAll(:,(j-1)*6+i)-mins(i))/maxs(i);
   PosFeaturesAll(:,(j-1)*6+i) = (PosFeaturesAll(:,(j-1)*6+i)-mins(i))/maxs(i);
   NegFeaturesAll(:,(j-1)*6+i) = (NegFeaturesAll(:,(j-1)*6+i)-mins(i))/maxs(i);
   end
end


%%
%save('PTM.mat','PosTrainMeans','PosTrainSTDs','PosTrainL','PosTrainS','PosTrainT');
%save('NTM.mat','NegTrainMeans','NegTrainSTDs','NegTrainL','NegTrainS','NegTrainT');
save('trainFeatures.mat','PosFeaturesAll','PosTrainLabel', 'NegFeaturesAll','NegTrainLabel');
save('testFeatures.mat','NegTestFeaturesAll','DNegTestFeaturesAll','PosTestFeaturesAll','DPosTestFeaturesAll','PosTestPredict','DPosTestPredict','NegTestPredict','DNegTestPredict');


%%
PosTestPredict = ones(size(PosTestFeaturesAll,1),1);
NegTestPredict = -1*ones(size(NegTestFeaturesAll,1),1);
DPosTestPredict = ones(size(DPosTestFeaturesAll,1),1);
DNegTestPredict = -1*ones(size(DNegTestFeaturesAll,1),1);


NegTrainLabel = NegTrainLabel(:,1);
PosTrainLabel = PosTrainLabel(:,1);

TrainingSet = vertcat(PosFeaturesAll, NegFeaturesAll);
TrainingLabel = vertcat(PosTrainLabel,NegTrainLabel);

TestingSet = vertcat(PosTestFeaturesAll,NegTestFeaturesAll);
DTestingSet = vertcat(DPosTestFeaturesAll,DNegTestFeaturesAll);

NegTestPredict = NegTestPredict(:,1);
PosTestPredict = PosTestPredict(:,1);
Predict = vertcat(PosTestPredict,NegTestPredict);

DNegTestPredict = DNegTestPredict(:,1);
DPosTestPredict = DPosTestPredict(:,1);
DPredict = vertcat(DPosTestPredict,DNegTestPredict);

net = svm(size(TrainingSet,2),'rbf',[0.5],5);

net = svmtrain(net,TrainingSet,TrainingLabel);
[y,y1] = svmfwd(net, TestingSet);
[Dy,Dy1] = svmfwd(net, DTestingSet);
for i = 1:21
TP(i) = size(find(y(:,i)==Predict & Predict == 1),1);
FN(i) = size(find(y(:,i) ~= Predict & Predict ==1),1);
FP(i) = size(find(y(:,i) ~=Predict & Predict ==-1),1);
TN(i) = size(find(y(:,i) == Predict & Predict == -1),1);

DTP(i) = size(find(Dy(:,i)==DPredict & DPredict == 1),1);
DFN(i) = size(find(Dy(:,i) ~= DPredict & DPredict ==1),1);
DFP(i) = size(find(Dy(:,i) ~=DPredict & DPredict ==-1),1);
DTN(i) = size(find(Dy(:,i) == DPredict & DPredict == -1),1);

TPrate(i) = TP(i)/(TP(i)+FN(i))*100;
FPrate(i) = FP(i)/(FP(i)+TN(i))*100;

DTPrate(i) = DTP(i)/(DTP(i)+DFN(i))*100;
DFPrate(i) = DFP(i)/(DFP(i)+DTN(i))*100;

end



%% Neural Network
disp('1');
x= TrainingSet;
x= transpose(x);
y = zeros(2,size(TrainingLabel,1));
y(1,1:size(PosTrainLabel,1))=1;
y(2,size(PosTrainLabel,1)+1:size(TrainingLabel,1))=1;
setdemorandstream(491282);
net = patternnet(30);
view(net);
[net, tr] = train(net,x,y);
nntraintool
plotperform(tr)
%%
testX = x;
testT = y;
testIndices = vec2ind(testY)
testY = net(testX);
plotconfusion(testT,testY)
%%
testX = transpose(TestingSet);
testT = zeros(2,size(Predict,1));

DtestX = transpose(DTestingSet);
DtestT = zeros(2,size(DPredict,1));
testT(1,1:size(PosTestPredict,1))=1;
testT(2,size(PosTestPredict,1)+1:size(Predict,1))=1;
DtestT(1,1:size(DPosTestPredict,1))=1;
DtestT(2,size(DPosTestPredict,1)+1:size(DPredict,1))=1;

testY = net(testX);
plotconfusion(testT,testY)
[c,cm] = confusion(testT,testY)
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

DtestY = net(DtestX);
%plotconfusion(DtestT,DtestY)
[Dc,Dcm] = confusion(DtestT,DtestY)
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-Dc));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*Dc);

%%
TNidx = find(y(:,11) == Predict & Predict == -1);
TPidx = find(y(:,11)==Predict & Predict == 1);
FNidx = find(y(:,11) ~= Predict & Predict ==1);
FPidx = find(y(:,11) ~=Predict & Predict ==-1);


