function [infor,labelId,Matchdate]= RecognitionEx(imagePath, adpath, pubdate, textCnn, digtCnn, boxList, labelsTable,boxads)
try
    infor.res = 'no';
    infor.inv = 0;   
    labelId = '0000000';
    Matchdate = pubdate;
    %% 加载要识别的原始图像块
%     colorImage = imread(imagePath);
    colorImage = imagePath;
    [height, width, ~] = size(colorImage);        
    if width>height
        return;
    end
    if height> 2592
        colorImage = imresize(colorImage, 2592/height);
    end
    Image_Width = width*2592/height;
    
    gray = rgb2gray(colorImage);
%     if strcmp(citycode,'731')
%         src = colorImage(1:uint32(height*0.33), :, 1);  
%     else
      src = gray(1:uint32(height*0.33), :);  
%     end
    mserRegions = detectMSERFeatures(src);
    if mserRegions.Count>6
        mserRegionsPixels = vertcat(cell2mat(mserRegions.PixelList));
        mask = false(size(src));
        ind = sub2ind(size(mask), mserRegionsPixels(:,2), mserRegionsPixels(:,1));
        mask(ind) = true;        
        
        bw = adaptiveThreshold(src, [25, 25], 3);       
        bin = mask & bw;        
        bin = selectShape(bin, [4, 100], [4, 100]);
        
    else
        return;
    end
    %% 膨胀       
    se = strel('rectangle', [11, 17]);
    res_d = imdilate(bin, se);
    L = bwlabeln(res_d, 8);
    S = regionprops(L, 'BoundingBox');
    %% 选择候选区域，并且统计块  
    labels = struct([]);
    for i = 1:size(S)
        block = L == i;
        rect = int32(S(i).BoundingBox);
        if rect(3)>40 && rect(4)>20 && rect(3)<500 && rect(4)<300
            block = block & bin;
            im = block(rect(2):rect(2)+rect(4)-1,rect(1):rect(1)+rect(3)-1);
            label = statsLabel(im, textCnn); 
            label.rect = rect;
            if label.nums<1
                continue;
            end
            labels = [labels, label];
        end
    end     
    if isempty(labels)
        return;
    end
    %% divide region     
    for i=1:length(labels)
        region = [labels(i).region];
        box = [region([region.text]==1).box];
        width = mean(box(3,:));
        index = 0;
        for j=1:length(region)
            multi = round(region(j).box(3)/width);
            if region(j).text==0 && multi>1 && multi<4 
                regions = divideRegion(region(j), multi, textCnn);                        
                labels(i).region = [labels(i).region(1:(j-1+index)), regions, labels(i).region((j+1+index):end)];
                labels(i).nums = labels(i).nums  + sum([regions.text]);
                index = index + multi-1;
            end            
        end 
    end   
    labelsLen = length(labels);
    for i=labelsLen:-1:1       
        center  = [labels(i).region([labels(i).region.text]==1).center];
        box     = [labels(i).region([labels(i).region.text]==1).box];
        if labels(i).nums == 7
            continue;
        end
        if size(center, 2) < 2
            labels(i)=[];
            continue;
        end       
        textSz = [mean(box(3, :)); mean(box(4, :))];  
        labels(i).textSize = textSz;             
        pcenter = polyfit(center(1, :), center(2, :), 1);
        pup = pcenter+[0, textSz(2)/2.0];
        pdown = pcenter-[0, textSz(2)/2.0];      
        for j= length(labels(i).region):-1:1
            if labels(i).region(j).text==0
                ct = labels(i).region(j).center;                
                if abs(polyval(pcenter, ct(1)) - ct(2)) < textSz(2)/2.0
                    labels(i).region(j).text=2;
                    labels(i).nums = labels(i).nums  + 1;
                    bx = ceil(labels(i).region(j).box);
                    if bx(4)>textSz(2)+1
                        for xx=bx(1):bx(1)+bx(3)                    
                           yt = round(polyval(pup, xx));
                           yd = round(polyval(pdown, xx));
                           labels(i).bw(min(max(1, yt), bx(2)+bx(4)):max(min(yt, labels(i).rect(4)), bx(2)+bx(4)), xx) = 0;
                           labels(i).bw(min(max(1, yd), bx(2)):max(min(yd, labels(i).rect(4)), bx(2)), xx) = 0;                           
                        end                         
                        bx = labels(i).region(j).box;
                        textArea =labels(i).bw(uint32(bx(2):bx(2)+bx(4)-1), uint32(bx(1):bx(1)+bx(3)-1))&labels(i).region(j).bw;   
                        rows = all(textArea==0, 2);
                        vertical = find(rows==0); 
                        cols = all(textArea==0, 1);
                        horizontal = find(cols==0);
                        if isempty(vertical)
                            labels(i).region(j) = [];
                            continue;
                        end
                        textArea = textArea(vertical(1):vertical(end), horizontal(1):horizontal(end));
                        if characterRecognition(textArea, textCnn)
                            labels(i).region(j).text = 1;
                        end
                        labels(i).region(j).bw = textArea;
                        labels(i).region(j).box = [bx(1)+horizontal(1)-1; bx(2)+vertical(1)-1; size(textArea, 2);size(textArea, 1)];
                        labels(i).region(j).center = [bx(1)+horizontal(1)-1; bx(2)+vertical(1)-1] + [size(textArea, 2);size(textArea, 1)]/2;                 
                    end      
                else
                    labels(i).region(j) = [];
                end    
            end
        end      
        labels(i).region = labels(i).region([labels(i).region.text]>0);        
        if length(labels(i).region)<7
            labels(i) = [];
            continue;
        end
        % 合并        
        jj = 1;
        while jj<length(labels(i).region)
            if labels(i).region(jj).text==2 && labels(i).region(jj+1).text==2
                if abs(labels(i).region(jj).center(1) - labels(i).region(jj+1).center(1)) < textSz(1)/4.0  
                    labels(i).region(jj).text = 2;
                    box1 = labels(i).region(jj+1).box;
                    box2 = labels(i).region(jj).box;
                    box1 = [box1(1), box1(2), box1(1)+box1(3), box1(2)+box1(4)];
                    box2 = [box2(1), box2(2), box2(1)+box2(3), box2(2)+box2(4)];
                    rect =[min(box1(1), box2(1)), min(box1(2), box2(2)), max(box1(3), box2(3)), max(box1(4), box2(4))];
                    
                    labels(i).region(jj).box = [rect(1);rect(2); rect(3)-rect(1); rect(4)-rect(2)];
                    labels(i).region(jj).bw = labels(i).bw(uint32(rect(2):rect(4)-1), uint32(rect(1):rect(3)-1));
                    if characterRecognition(labels(i).region(jj).bw, textCnn)
                        labels(i).region(jj).text = 1;
                    end
                    labels(i).nums = labels(i).nums - 1; 
                    labels(i).region(jj+1)=[];
                end
            end
            jj = jj+1;
        end 
    end    
   num_labels = length(labels);
    for i=1:length(labels)   
        labels(i).weights = 0;
        label = labels(i);
        if label.nums > 7
            noText = find([label.region.text]);
            box     = [label.region(noText).box];
            sz = box(3:4, :)- repmat(label.textSize, [1, size(box, 2)]);
            [~, idx] = sort(sum(sz.^2, 1));
            index = false(1, numel(label.region));
            index(noText(idx(8:end))) = true;
            labels(i).region(index) = [];           
            labels(i).nums = 7;
        end       
        if  labels(i).nums == 7          
           [digt, weight] = LabelRecognition(labels(i), digtCnn,num_labels);
           labels(i).digt = digt;
           labels(i).weight = weight;
           labels(i).weights = prod(weight);  
        end
    end    
    if isempty(labels)
        return;
    end        
    [M, I] = max([labels.weights]);
    if M>0     
        labelId = char(labels(I).digt+'0');
        infor = table2struct(labelsTable(labelsTable.id==str2double(labelId), :));
        % figure,imshow(gray)
        label_y1 = labels(I).rect(1);
        label_y2 = labels(I).rect(3)/2;
        label_Y = (label_y1+label_y2);
        image_width_Left = Image_Width/2 - 10;
        image_width_Right = Image_Width/2 + 10;
         if ~isempty(infor) && infor.type<5  
             ad = infor.ad;
             ads = boxads(:,1);
             [~,ins]=ismember(ad,ads);
             isim = cell2mat(boxads(ins,2));
            if isim == 0
                [tform,~, idx, infor.inv, infor.score,validnums,imge] = checkLabelId(gray, boxList(infor.ad)); 
                ad = infor.ad;
                box = boxList(infor.ad);
                if infor.inv==1
                    infor.res = 'check';
                elseif infor.score==0&&validnums>4        
                    [tform,sc] = recheckT(imge,tform,box,ad,adpath,idx);
                    if isempty(tform)
                        infor.res = 'check';                   
                    else
                        infor.score = sc;
                        infor.res = 'ok';                   
                        if label_Y<image_width_Left                   
                            infor.pos = '0';
                        elseif label_Y>image_width_Right
                            infor.pos = '2';
                        else                   
                            infor.pos = '1';
                        end
                    end
                elseif infor.score~=0
                    infor.score = infor.score;
                    infor.res = 'ok';           
                    if label_Y<image_width_Left                   
                        infor.pos = '0';
                    elseif label_Y>image_width_Right
                            infor.pos = '2';
                    else                   
                            infor.pos = '1';
                    end
                else
                    infor = [];
                    infor.id = str2double(labelId);
                    infor.res = 'check';
                end
            else
                [~,~, ~, infor.inv, infor.score,validnums,imge] = checkLabelId(gray, boxList(infor.ad)); 
                ad = infor.ad;
                box = boxList(infor.ad);
                if infor.inv==1
                    infor.res = 'check';
                elseif infor.score==0&&validnums>4        
                     sc = Simrecheck(imge,box,ad,adpath);
                     if sc ==0
                        infor.res = 'check';                   
                     else
                        infor.score = sc;
                        infor.res = 'ok';                   
                        if label_Y<image_width_Left                   
                            infor.pos = '0';
                        elseif label_Y>image_width_Right
                            infor.pos = '2';
                        else                   
                            infor.pos = '1';
                        end
                     end
                elseif infor.score~=0
                    infor.score = infor.score;
                    infor.res = 'ok';           
                    if label_Y<image_width_Left                   
                        infor.pos = '0';
                    elseif label_Y>image_width_Right
                            infor.pos = '2';
                    else                   
                            infor.pos = '1';
                    end
                else
                 infor = [];
                 infor.id = str2double(labelId);
                 infor.res = 'check';
                end
            end
        elseif ~isempty(infor) && ~strcmp(infor.pubdate, pubdate)
            imagedir = fullfile(adpath, 'DesignImagesList', infor.ad);
            if ~exist(imagedir, 'dir')
                infor.res = 'blankcheck';
            else                
                tform = checkLabelId(gray, readAdImage(imagedir));
                if isempty(tform)
                    infor.res = 'blankcheck';
                else
                    infor.res = 'blankok';
                end
            end
        else
            infor = [];
            infor.id = str2double(labelId);
            infor.res = 'check';
        end   
    end
  catch err 
    infor.res = 'error';
    fprintf('%s\n%s\nline:%d\n',err.message,err.stack.file,err.stack.line);
  end
end


function [Tform,score] = recheckT(imge,tform,box,ad,adpath,idx)
Tform = [];
score = 0;
Match = 0;
warning('off');
try
    if isempty(tform)
        return;
    else          
%     filename = fullfile(adpath,  'design', ad);
    filename = fullfile(adpath,  'DesignImagesList', ad);
    imgs = dir([filename, '/*.jpg']);
    for j = 1:numel(imgs)
        image = imread(fullfile(filename, imgs(j).name));             
        original = rgb2gray(imresize(image, [960,720]));  
        ptsOriginal  = detectSURFFeatures(original);
        ptsDistorted = detectSURFFeatures(imge);
        [featuresOriginal,   validPtsOriginal] = extractFeatures(original,  ptsOriginal);
        [featuresDistorted, validPtsDistorted] = extractFeatures(imge, ptsDistorted);
        index_pairs = matchFeatures(featuresOriginal, featuresDistorted);
        matchedPtsOriginal  = validPtsOriginal(index_pairs(:,1));
        matchedPtsDistorted = validPtsDistorted(index_pairs(:,2));
        [tform,~,~] = estimateGeometricTransform(matchedPtsDistorted,matchedPtsOriginal,'projective');
        outputView = imref2d([960,720]);
        distorted = imwarp(imge, tform,'OutputView',outputView);
%         figure,imshow(distorted)
        ptsOriginal  = detectSURFFeatures(original);
        ptsDistorted = detectSURFFeatures(distorted);
        [featuresOriginal, validPtsOriginal] = extractFeatures(original, ptsOriginal);
        [featuresDistorted, validPtsDistorted] = extractFeatures(distorted, ptsDistorted);
        [index_pairs,matchmetric] = matchFeatures(featuresOriginal, featuresDistorted);
        validnums = size(index_pairs, 1);
        if validnums < 4
            Tform = [];
            continue;
        end
        matchedPtsOriginal  = validPtsOriginal(index_pairs(:,1));
        matchedPtsDistorted = validPtsDistorted(index_pairs(:,2));
        try
            Tform  = estimateGeometricTransform(matchedPtsOriginal,matchedPtsDistorted,'projective');
        catch
            Tform = [];
        end
        T=Tform.T;
        rate = length(matchmetric)/length(box.p{idx});         
        if rate<0.10 && (T(1,1)<0.2 || T(1, 1)>2 || T(2,2)<0.2 || T(2, 2)>2 ...
            || T(3,1)>30 || T(3, 1)<-20|| T(3,2)>30 || T(3, 2)< -20)   %T(3, 2)<0 
            Tform = []; 
        else
            Match = 1;
            break;
        end
    end
    if ~isempty(Tform) && Match == 1
        w = box.sz(2);
        h = box.sz(1);
        boxPolygon = [1, 1;...             % top-left
                w, 1;...                 % top-right
                w, h;...               % bottom-right
                1, h;...                 % bottom-left
                1, 1];                     % top-left again to close the polygon
        X = transformPointsForward(Tform, boxPolygon);
        mask = poly2mask(double(X(:, 1)), double(X(:, 2)), size(imge, 1),size(imge, 2));
        imge(mask) = 0;
        mserRegions = detectMSERFeatures(imge);
        score = validnums/mserRegions.Count;
    end
    end
catch err
    disp(err.message);
end
end



function score =  Simrecheck(imge,box,ad,adpath)
score=0;
warning('off');
try
%     if isempty(tform)
%         return;
%     else          
%     filename = fullfile(adpath,  'design', ad);
    filename = fullfile(adpath,  'DesignImagesList', ad);
    imgs = dir([filename, '/*.jpg']);
    for j = 1:numel(imgs)
        image = imread(fullfile(filename, imgs(j).name));             
        original = rgb2gray(imresize(image, [960,720]));  
%         figure,imshow(original)ptsOriginal  = detectKAZEFeatures(original);
%         ptsDistorted = detectKAZEFeatures(imge);
        ptsOriginal  = detectSURFFeatures(original);
        ptsDistorted = detectSURFFeatures(imge);
        [featuresOriginal,   validPtsOriginal] = extractFeatures(original,  ptsOriginal);
        [featuresDistorted, validPtsDistorted] = extractFeatures(imge, ptsDistorted);
        index_pairs = matchFeatures(featuresOriginal, featuresDistorted);
        matchedPtsOriginal  = validPtsOriginal(index_pairs(:,1));
        matchedPtsDistorted = validPtsDistorted(index_pairs(:,2));
        validnums = size(index_pairs, 1);
        [tform,~,~] = estimateGeometricTransform(matchedPtsDistorted,matchedPtsOriginal,'projective');
        outputView = imref2d([960,720]);
        Ir = imwarp(imge, tform,'OutputView',outputView);
        distorted = Ir(50:919,50:669);
%         figure,imshow(distorted)
        sim= Calculat_diff(distorted,original);
        if sim < 0.085
            score = 0;
            break;
        else
             w = box.sz(2);
             h = box.sz(1);
             boxPolygon = [1, 1;...             % top-left
                    w, 1;...                 % top-right
                    w, h;...               % bottom-right
                    1, h;...                 % bottom-left
                    1, 1];                     % top-left again to close the polygon
            X = transformPointsForward(tform, boxPolygon);
            mask = poly2mask(double(X(:, 1)), double(X(:, 2)), size(imge, 1),size(imge, 2));
            imge(mask) = 0;
            mserRegions = detectMSERFeatures(imge);
            score = validnums/mserRegions.Count;
            break;
        end
    end
%     end
catch err
    disp(err.message);
end
end













