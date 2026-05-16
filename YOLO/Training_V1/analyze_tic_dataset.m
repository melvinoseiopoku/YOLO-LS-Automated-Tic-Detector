function analyze_tic_dataset()
% Publication-style dataset analysis for Label Studio -> YOLO video dataset
% Assumes this script lives in the same folder as:
%   - training.json
%   - yd/ (or change yoloDir below)
%
% Outputs:
%   - figures saved to ./dataset_report_figs/
%   - tables saved to ./dataset_report_tables/

%% -------------------- USER SETTINGS --------------------
baseDir = fileparts(mfilename('fullpath'));
jsonPath = fullfile(baseDir, "training.json");

yoloDir  = fullfile(baseDir, "yd");          % change if needed
fps      = 30;                               % LS framerate you used
targetVideoNums = [37 41 44 45 46];          % change if needed
saveFigs = true;
saveTables = true;
outFigDir = fullfile(baseDir, "dataset_report_figs");
outTabDir = fullfile(baseDir, "dataset_report_tables");

rng(0);

%% -------------------- LOAD + BASIC VALIDATION --------------------
assert(isfile(jsonPath), "Missing training.json at: %s", jsonPath);
assert(isfolder(yoloDir), "Missing YOLO dir at: %s", yoloDir);

if ~isfolder(outFigDir), mkdir(outFigDir); end
if ~isfolder(outTabDir), mkdir(outTabDir); end

tasks = jsondecode(fileread(jsonPath));
if iscell(tasks), tasks = [tasks{:}]; end

% Filter tasks to selected videos (MAH00037 etc.)
[tasksF, meta] = filterTasks(tasks, targetVideoNums);

fprintf("Found %d tasks for target videos.\n", numel(tasksF));
disp(meta(:,["video","framesCount","durationSec"]))

%% -------------------- PARSE BOXES FROM LABEL STUDIO (FRAME-LEVEL) --------------------
% We expand each videorectangle track into per-frame boxes based on enabled intervals
% and interpolate box geometry between adjacent enabled events.

[boxTbl, perVideoTbl, classNames] = parseLSBoxes(tasksF, fps);

% boxTbl columns:
% video, split(unknown here), frame, timeSec, class, cx, cy, w, h, area, aspect
% perVideoTbl columns:
% video, framesCount, durationSec, labeledFrames, labeledFrac, nBoxes, boxesPerLabeledFrame, boxesPerAllFrames

disp(perVideoTbl);

%% -------------------- READ YOLO SPLITS (train/val/test) --------------------
% Infer which video went to which split based on filenames in yd/images/*
splitMap = inferVideoSplitsFromYOLO(yoloDir);

% Attach split to tables
boxTbl.split = repmat("unknown", height(boxTbl), 1);
for i = 1:height(boxTbl)
    v = boxTbl.video(i);
    if isKey(splitMap, char(v))
        boxTbl.split(i) = string(splitMap(char(v)));
    end
end

perVideoTbl.split = repmat("unknown", height(perVideoTbl), 1);
for i = 1:height(perVideoTbl)
    v = perVideoTbl.video(i);
    if isKey(splitMap, char(v))
        perVideoTbl.split(i) = string(splitMap(char(v)));
    end
end

%% -------------------- SUMMARY TABLES (ROBUST) --------------------
% (No groupsummary/groupcounts; uses findgroups+splitapply)

% Overall class counts (number of boxes)
[G1, cls1] = findgroups(string(boxTbl.class));
nBoxes1 = splitapply(@numel, boxTbl.frame, G1);
classCountsAll = table(cls1, nBoxes1, 'VariableNames', {'class','GroupCount'});

% Class counts by split
[G2, sp2, cls2] = findgroups(string(boxTbl.split), string(boxTbl.class));
nBoxes2 = splitapply(@numel, boxTbl.frame, G2);
classCountsSplit = table(sp2, cls2, nBoxes2, 'VariableNames', {'split','class','nBoxes'});

% Class counts by video
[G3, vid3, cls3] = findgroups(string(boxTbl.video), string(boxTbl.class));
nBoxes3 = splitapply(@numel, boxTbl.frame, G3);
videoClassCounts = table(vid3, cls3, nBoxes3, 'VariableNames', {'video','class','nBoxes'});

%% -------------------- FIGURE 1: CLASS COUNTS BY SPLIT (STACKED BAR) --------------------
figure('Color','w'); 
plotClassCountsBySplit(classCountsSplit, classNames);
if saveFigs, exportgraphics(gcf, fullfile(outFigDir,"fig1_class_counts_by_split.png"), 'Resolution', 300); end

%% -------------------- FIGURE 2: TIC TIMELINES PER VIDEO --------------------
% A raster/timeline: each dot = labeled frame occurrence (per class)
figure('Color','w');
plotTimelines(boxTbl, perVideoTbl);
if saveFigs, exportgraphics(gcf, fullfile(outFigDir,"fig2_timelines.png"), 'Resolution', 300); end

%% -------------------- FIGURE 3: BOX SIZE DISTRIBUTIONS --------------------
figure('Color','w');
plotBoxDistributions(boxTbl);
if saveFigs, exportgraphics(gcf, fullfile(outFigDir,"fig3_box_geometry.png"), 'Resolution', 300); end

%% -------------------- FIGURE 4: FRAMES vs LABELED FRAMES PER VIDEO --------------------
figure('Color','w');
plotVideoFrameStats(perVideoTbl);
if saveFigs, exportgraphics(gcf, fullfile(outFigDir,"fig4_video_coverage.png"), 'Resolution', 300); end

%% -------------------- OPTIONAL: EXAMPLE FRAMES WITH BOXES --------------------
% This pulls a handful of labeled frames from YOLO val split if available
try
    figure('Color','w');
    showExampleFrames(yoloDir, boxTbl, fps, 6);
    if saveFigs, exportgraphics(gcf, fullfile(outFigDir,"fig5_example_frames.png"), 'Resolution', 300); end
catch ME
    fprintf("Skipping example frames figure: %s\n", ME.message);
end

fprintf("Done. Figures: %s | Tables: %s\n", outFigDir, outTabDir);

end

%% ======================== HELPERS ========================

function [tasksF, meta] = filterTasks(tasks, targetNums)
% Keep tasks whose file_upload tail is MAH000##.MP4 matching targetNums
keep = false(1,numel(tasks));
video = strings(numel(tasks),1);
framesCount = nan(numel(tasks),1);
durationSec = nan(numel(tasks),1);

for i=1:numel(tasks)
    if ~isfield(tasks(i),'file_upload') || isempty(tasks(i).file_upload), continue; end
    tail = stripHashPrefix(string(tasks(i).file_upload)); % MAH00041.MP4
    num = parseMahNumber(tail);
    if ismember(num, targetNums)
        keep(i) = true;
        video(i) = erase(tail, ".MP4");
        [fc, dur] = getFramesAndDuration(tasks(i));
        framesCount(i) = fc;
        durationSec(i) = dur;
    end
end

tasksF = tasks(keep);
meta = table(video(keep), framesCount(keep), durationSec(keep), ...
    'VariableNames',["video","framesCount","durationSec"]);
end

function [boxTbl, perVideoTbl, classNames] = parseLSBoxes(tasks, fps)
% Expand videorectangle sequences into per-frame boxes.
boxRows = {};
classSet = strings(0,1);

pv_video = strings(numel(tasks),1);
pv_frames = nan(numel(tasks),1);
pv_dur = nan(numel(tasks),1);
pv_labeledFrames = nan(numel(tasks),1);
pv_nBoxes = nan(numel(tasks),1);

row = 0;

for ti=1:numel(tasks)
    tail = stripHashPrefix(string(tasks(ti).file_upload));
    vid = erase(tail, ".MP4");
    [framesCount, dur] = getFramesAndDuration(tasks(ti));
    pv_video(ti) = vid; pv_frames(ti) = framesCount; pv_dur(ti) = dur;

    labelsPerFrame = cell(framesCount,1);
    classesPerFrame = cell(framesCount,1);

    if isfield(tasks(ti),'annotations') && ~isempty(tasks(ti).annotations)
        anns = tasks(ti).annotations;
        for ai=1:numel(anns)
            if ~isfield(anns(ai),'result') || isempty(anns(ai).result), continue; end
            res = anns(ai).result;
            for ri=1:numel(res)
                if ~isfield(res(ri),'type') || ~strcmp(res(ri).type,'videorectangle'), continue; end
                if ~isfield(res(ri),'value'), continue; end
                val = res(ri).value;
                if ~isfield(val,'labels') || isempty(val.labels), continue; end
                lab = string(val.labels{1});
                classSet(end+1,1) = lab; %#ok<AGROW>

                if ~isfield(val,'sequence') || isempty(val.sequence), continue; end
                seq = val.sequence;
                [labelsPerFrame, classesPerFrame] = addTrack(labelsPerFrame, classesPerFrame, seq, lab, framesCount);
            end
        end
    end

    % Build row table
    nBoxes = 0;
    labeled = 0;
    for f=1:framesCount
        if ~isempty(labelsPerFrame{f})
            labeled = labeled + 1;
            boxes = labelsPerFrame{f};
            labs  = classesPerFrame{f};
            for k=1:numel(boxes)
                nBoxes = nBoxes + 1;
                row = row + 1;
                b = boxes{k}; % [cx cy w h] normalized
                area = b(3)*b(4);
                aspect = b(3)/max(b(4), eps);
                boxRows(row,:) = {vid, f, (f-1)/fps, labs(k), b(1), b(2), b(3), b(4), area, aspect}; %#ok<AGROW>
            end
        end
    end

    pv_labeledFrames(ti) = labeled;
    pv_nBoxes(ti) = nBoxes;
end

boxTbl = cell2table(boxRows, ...
    'VariableNames',["video","frame","timeSec","class","cx","cy","w","h","area","aspect"]);

% per-video stats
perVideoTbl = table(pv_video, pv_frames, pv_dur, pv_labeledFrames, ...
    pv_labeledFrames./pv_frames, pv_nBoxes, ...
    pv_nBoxes./max(pv_labeledFrames,1), pv_nBoxes./max(pv_frames,1), ...
    'VariableNames',["video","framesCount","durationSec","labeledFrames","labeledFrac","nBoxes","boxesPerLabeledFrame","boxesPerAllFrames"]);

classNames = unique(classSet, 'stable');
end

function [labelsPerFrame, classesPerFrame] = addTrack(labelsPerFrame, classesPerFrame, seq, lab, framesCount)
% Expand sequence into enabled intervals. Interpolate geometry across adjacent enabled keyframes.

if iscell(seq), seq = [seq{:}]; end
hasFrame = arrayfun(@(s) isfield(s,'frame') && ~isempty(s.frame), seq);
seq = seq(hasFrame);
if isempty(seq), return; end

frames = arrayfun(@(s) double(s.frame), seq);
[frames, idx] = sort(frames);
seq = seq(idx);

for i=1:numel(seq)
    f1 = double(seq(i).frame);
    if f1 < 1 || f1 > framesCount, continue; end

    en = true;
    if isfield(seq(i),'enabled') && ~isempty(seq(i).enabled)
        en = logical(seq(i).enabled);
    end
    if ~en, continue; end

    if ~all(isfield(seq(i), {'x','y','width','height'})), continue; end
    x1 = double(seq(i).x); y1 = double(seq(i).y);
    w1 = double(seq(i).width); h1 = double(seq(i).height);

    if i < numel(seq)
        f2 = double(seq(i+1).frame) - 1;
    else
        f2 = framesCount;
    end
    f2 = min(max(f2, f1), framesCount);

    % Interpolate end geometry if next is enabled and has geom
    x2=x1; y2=y1; w2=w1; h2=h1;
    if i < numel(seq) && isfield(seq(i+1),'enabled') && logical(seq(i+1).enabled) ...
            && all(isfield(seq(i+1), {'x','y','width','height'}))
        x2 = double(seq(i+1).x); y2 = double(seq(i+1).y);
        w2 = double(seq(i+1).width); h2 = double(seq(i+1).height);
    end

    span = f2 - f1;
    for f=f1:f2
        t = 0; if span>0, t = (f-f1)/span; end
        x = x1 + t*(x2-x1);
        y = y1 + t*(y2-y1);
        w = w1 + t*(w2-w1);
        h = h1 + t*(h2-h1);

        % Convert LS percent top-left to YOLO normalized center
        [cx, cy, ww, hh] = ls2yolo(x,y,w,h);
        labelsPerFrame{f}{end+1} = [cx cy ww hh]; %#ok<AGROW>
        classesPerFrame{f}(end+1) = lab; %#ok<AGROW>
    end
end
end

function [cx,cy,ww,hh] = ls2yolo(xPct,yPct,wPct,hPct)
x0 = xPct/100; y0 = yPct/100; ww = wPct/100; hh = hPct/100;
cx = x0 + ww/2; cy = y0 + hh/2;
cx = min(max(cx,0),1); cy = min(max(cy,0),1);
ww = min(max(ww,0),1); hh = min(max(hh,0),1);
end

function plotClassCountsBySplit(classCountsSplit, classNames)
% Make pivot table: split x class
splits = unique(string(classCountsSplit.split), 'stable');
C = zeros(numel(splits), numel(classNames));
for i=1:height(classCountsSplit)
    s = string(classCountsSplit.split(i));
    c = string(classCountsSplit.class(i));
    n = classCountsSplit.nBoxes(i);
    si = find(splits==s);
    ci = find(classNames==c);
    if ~isempty(si) && ~isempty(ci)
        C(si,ci) = n;
    end
end
b = bar(C, 'stacked'); %#ok<NASGU>
xlabel("Split"); ylabel("Number of Boxes");
set(gca,'XTickLabel',splits);
legend(classNames,'Location','bestoutside');
box off;
end

function plotTimelines(boxTbl, perVideoTbl)
vids = perVideoTbl.video;
hold on;
for i=1:numel(vids)
    v = vids(i);
    sub = boxTbl(boxTbl.video==v,:);
    % Jitter by class a tiny bit for readability
    y = i + 0.12*randn(height(sub),1);
    scatter(sub.timeSec, y, 4, 'filled'); %#ok<*UNRCH>
end
yticks(1:numel(vids)); yticklabels(vids);
xlabel("Time (s)"); ylabel("Video");
box off;
end

function plotBoxDistributions(boxTbl)
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
histogram(boxTbl.area, 50);
xlabel("Box Area (normalized)"); ylabel("Count");
box off;

nexttile;
histogram(boxTbl.aspect, 50);
xlabel("Aspect Ratio (w/h)"); ylabel("Count");
box off;
end

function plotVideoFrameStats(perVideoTbl)
% Bars: total frames vs labeled frames
vids = perVideoTbl.video;
x = 1:numel(vids);
bar(x, [perVideoTbl.framesCount, perVideoTbl.labeledFrames], 'grouped');
set(gca,'XTick',x,'XTickLabel',vids);
xtickangle(25);
ylabel("Frames");
legend(["Total frames","Labeled frames"], 'Location','best');
box off;
end

function showExampleFrames(yoloDir, boxTbl, fps, n)
% Pull random labeled frames from val split if possible
imgDir = fullfile(yoloDir,"images","val");
if ~isfolder(imgDir)
    imgDir = fullfile(yoloDir,"images","train");
end
imgs = dir(fullfile(imgDir,"*.jpg"));
assert(~isempty(imgs), "No images found in %s", imgDir);

% Prefer frames that have boxes
labeled = unique(boxTbl(:,["video","frame"]));
idx = randperm(height(labeled), min(n, height(labeled)));

tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
for i=1:numel(idx)
    v = labeled.video(idx(i));
    f = labeled.frame(idx(i));
    % filename format in your exporter: MAH00041_f0024165.jpg (LS frame index)
    pat = sprintf("%s_f%06d.jpg", v, f);
    p = fullfile(imgDir, pat);
    if ~isfile(p)
        nexttile; axis off; title("Missing: "+pat,'Interpreter','none');
        continue
    end

    I = imread(p);
    nexttile; imshow(I); hold on; title(pat,'Interpreter','none');

    sub = boxTbl(boxTbl.video==v & boxTbl.frame==f,:);
    [H,W,~] = size(I);
    for k=1:height(sub)
        cx=sub.cx(k); cy=sub.cy(k); ww=sub.w(k); hh=sub.h(k);
        x=(cx-ww/2)*W; y=(cy-hh/2)*H;
        rectangle('Position',[x y ww*W hh*H], 'LineWidth', 1.5);
    end
end
end

function splitMap = inferVideoSplitsFromYOLO(yoloDir)
% map video stem (MAH00041) -> split (train/val/test)
splitMap = containers.Map('KeyType','char','ValueType','char');
splits = ["train","val","test"];
for s = splits
    imgDir = fullfile(yoloDir,"images",s);
    if ~isfolder(imgDir), continue; end
    imgs = dir(fullfile(imgDir,"*.jpg"));
    for i=1:numel(imgs)
        name = string(imgs(i).name);
        % stem is before "_f"
        parts = split(name, "_f");
        if numel(parts) >= 2
            stem = parts(1);
            if ~isKey(splitMap, char(stem))
                splitMap(char(stem)) = char(s);
            end
        end
    end
end
end

function [framesCount, durationSec] = getFramesAndDuration(task)
framesCount = NaN;
durationSec = NaN;
if ~isfield(task,'annotations') || isempty(task.annotations), return; end
anns = task.annotations;
for ai=1:numel(anns)
    if ~isfield(anns(ai),'result') || isempty(anns(ai).result), continue; end
    res = anns(ai).result;
    for ri=1:numel(res)
        if ~isfield(res(ri),'type') || ~strcmp(res(ri).type,'videorectangle'), continue; end
        val = res(ri).value;
        if isfield(val,'framesCount') && ~isempty(val.framesCount)
            framesCount = double(val.framesCount);
        end
        if isfield(val,'duration') && ~isempty(val.duration)
            durationSec = double(val.duration);
        end
    end
end
end

function s = stripHashPrefix(fileUpload)
txt = char(fileUpload);
dash = find(txt=='-', 1, 'last');
if ~isempty(dash) && dash < numel(txt)
    s = string(txt(dash+1:end));
else
    s = string(txt);
end
end

function num = parseMahNumber(tail)
num = NaN;
m = regexp(char(tail), 'MAH0*([0-9]+)\.MP4', 'tokens', 'once', 'ignorecase');
if ~isempty(m)
    num = str2double(m{1});
end
end

function s = setDefault(s, field, value)
if ~isfield(s, field) || isempty(s.(field))
    s.(field) = value;
end
end