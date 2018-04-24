function [avgPrecision,result,tblPrecsionRecall] = predictOnTestDataset(detector,testData,saveModelPath)
    fprintf('Evaluating for Test set...\n');
    resultsStruct = struct([]);
    savedTestImgPath=strcat(saveModelPath,'/test_img');
    mkdir(savedTestImgPath);
    negImgInfo=strcat(savedTestImgPath,'/neg_img_info.txt');
    negfid = fopen(negImgInfo, 'w+');    
    for i = 1:height(testData)      
        filename=testData.filename{i};
        I = imread(filename);        
        [bboxes, scores, labels] = detect(detector, I);        
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;        
        % save image after creating bounding box
        saveImage(I,bboxes,scores,filename,savedTestImgPath)
        if size(bboxes,1) ==0
            sfn=split(filename,'/');
            fprintf('** NO FACE FOUND:%s/%s\n',sfn{end-1},sfn{end});
            fprintf(negfid,'NO FACE FOUND:%s/%s\n',sfn{end-1},sfn{end});
            
        end
    end  
    fclose(negfid);
    % Convert the results into a table.
    result = struct2table(resultsStruct);
    savepath=strcat(saveModelPath,'/','test_result.mat');
    save(savepath,'result');
    fprintf('Completed..\n');
    
    fprintf('Evaluate the object detector using Average Precision metric...\n');
    expectedResults = testData(:, 2:end);
    [avgPrecision, recall, precision] = evaluateDetectionPrecision(result, expectedResults);    
    tbl=table(precision,recall);
    savepath=strcat(saveModelPath,'/','precision_recall.mat');    
    save(savepath,'tbl');
    fprintf('Completed\n');    
    fprintf('**Avg Precision of Dectector:%f\n',avgPrecision);
    tblPrecsionRecall=tbl;    
end

% Drawing box and Saving Image    
function saveImage(img,bboxes,scores,filename,savedTestImgPath)
        sfn=split(filename,'/');
        folder=strcat(savedTestImgPath,'/',sfn{end-1});  
        if ~(exist(folder))
            mkdir(folder);
        end
        if size(bboxes,1)>0
            boxImg = insertObjectAnnotation(img,'rectangle',bboxes,scores);        
        else            
            boxImg=img;
        end
        imwrite(boxImg,strcat(folder,'/',sfn{end}));        
end

