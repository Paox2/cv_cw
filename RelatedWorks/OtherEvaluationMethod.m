

conMatrix = confusionmat(testLabel, outputID);

accPerLabel = diag(conMatrix) ./ sum(conMatrix,2);

TPList = diag(conMatrix);
FPList = [];
FNList = [];
for i = 1 : 100
    FPList(i) = sum(conMatrix(:,i))-TPList(i);
    FNList(i) = sum(conMatrix(i,:))-TPList(i);
end

preList = [];
recList = [];
sumPre = 0.0;
sumRec = 0.0;
sumF1 = 0.0;

for i = 1 : 100
    preList(i) = TPList(i) / (TPList(i) + FPList(i));
    recList(i) = TPList(i) / (TPList(i) + FNList(i));
    if preList(i) ~= 0 || recList(i) ~= 0
        sumF1 = sumF1 + 2 * preList(i) * recList(i) / (preList(i) + recList(i));
    end
end

sumPre = sum(preList);
sumRec = sum(recList);
zpre = sumPre;
zrec = sumRec;
zf1 = sumF1;


