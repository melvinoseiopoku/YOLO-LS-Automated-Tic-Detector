%Sanity check

base = fullfile(pwd, "yd");  % if you are in Training_V1
splits = ["train","val","test"];
for s = splits
    imgDir = fullfile(base,"images",s);
    labDir = fullfile(base,"labels",s);
    nImg = numel(dir(fullfile(imgDir,"*.jpg")));
    nLab = numel(dir(fullfile(labDir,"*.txt")));
    fprintf("%s: images=%d, labels=%d\n", s, nImg, nLab);
end

base = fullfile(pwd,"yd");
labDir = fullfile(base,"labels","train");
imgDir = fullfile(base,"images","train");

labs = dir(fullfile(labDir,"*.txt"));
rng(0);
pick = randperm(numel(labs), min(20,numel(labs)));

for i = 1:numel(pick)
    labPath = fullfile(labDir, labs(pick(i)).name);
    imgPath = fullfile(imgDir, replace(labs(pick(i)).name,".txt",".jpg"));
    if ~isfile(imgPath), continue; end

    I = imread(imgPath);
    figure(1); clf; imshow(I); hold on; title(labs(pick(i)).name, 'Interpreter','none');

    lines = string(splitlines(string(fileread(labPath))));
    lines(lines=="") = [];
    for k = 1:numel(lines)
        parts = split(strtrim(lines(k)));
        cx = str2double(parts(2)); cy = str2double(parts(3));
        w  = str2double(parts(4)); h  = str2double(parts(5));

        [H,W,~] = size(I);
        x = (cx - w/2)*W; y = (cy - h/2)*H;
        rectangle('Position',[x y w*W h*H],'EdgeColor','g','LineWidth',2);
    end
    drawnow; pause(0.4);
end
