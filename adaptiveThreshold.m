function bw = adaptiveThreshold(gray, size, t)
h = fspecial('average', size);%��ֵ�˲�
% h = fspecial('log', size);%��ֵ�˲�
mean =uint8(filter2(h, gray));    
bw = gray < mean-t;  
end