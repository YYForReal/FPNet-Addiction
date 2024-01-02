% This example demonstrates how to use the MiPOD embedding function
clc
clear all
close all

% Read the input cover image
Cover = double(imread ('1.pgm'));

% Set the payload to 0.4 bpp
Payload = 0.4;

% MiPOD embedding
tStart = tic;
[Stego, pChange, ChangeRate] = MiPOD( Cover, Payload );
tEnd = toc(tStart);
fprintf('MiPOD embedding is done in: %f (sec)\n',tEnd);

%%
close all

figure;
imshow (Cover,[]);
title ('Cover image');

figure;
imshow(1-pChange/0.3333);
title('MiPOD - Embedding Change Probabilities');

figure;
imshow(Stego-Cover,[]);
title('MiPOD - Changed Pixels (+1 -> white ,-1 -> black)');


imwrite(Stego, 'stego_image.png'); % 将Stego保存为名为stego_image.png的PNG图像文件
% imwrite(uint8(Stego), 'stego_image.pgm', 'pgm'); % 将Stego保存为名为stego_image.pgm的PGM图像文件
% 将图像缩放到 [0, 255] 范围内
% Stego_scaled = mat2gray(Stego);

% 将图像保存为 PGM 文件
% imwrite(Stego_scaled, 'stego_image.pgm');

% 首先，交换行列以匹配 PGM 文件格式
Stego = Stego';


% 首先，打开一个文件用于写入
fid = fopen('stego_image.pgm', 'wb');

% 写入 PGM 头信息
fprintf(fid, 'P5\n');
fprintf(fid, '%d %d\n', size(Stego, 2), size(Stego, 1));
fprintf(fid, '255\n');

% 将图像数据以二进制形式写入
fwrite(fid, Stego, 'uint8');

% 关闭文件
fclose(fid);

