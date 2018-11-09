function bw = adaptiveThreshold(gray, size, t)
h = fspecial('average', size);%均值滤波
% h = fspecial('log', size);%均值滤波
mean =uint8(filter2(h, gray));    
bw = gray < mean-t;  
end