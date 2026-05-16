function ls_videorectangle_to_yolo_allframes(exportJsonName, opts)
% Label Studio VideoRectangle JSON -> Ultralytics YOLO detection dataset
% Exports ALL frames (optionally strided) and expands LS sparse "sequence"
% into per-frame labels using enabled/disabled intervals.
%
% Folder layout (same folder as this .m file):
%   ./training.json
%   ./videos/MAH00037.MP4 ...
%
% Output:
%   ./<outDirName>/images/{train,val,test}
%   ./<outDirName>/labels/{train,val,test}
%   ./<outDirName>/dataset.yaml
%
% Usage:
%   opts = struct();
%   opts.targetVideoNums = [37 41 44 45 46];
%   opts.frameStride = 1; % ALL frames (can set to 5, 10 to reduce size)
%   ls_videorectangle_to_yolo_allframes("training.json", opts);

if nargin < 2 || isempty(opts), opts = struct(); end

opts = setDefault(opts, 'videosSubdir', "videos");
opts = setDefault(opts, 'outDirName',  "yolo_dataset");
opts = setDefault(opts, 'seed', 0);
opts = setDefault(opts, 'splitRatios', [0.60 0.20 0.20]);  % by VIDEO
opts = setDefault(opts, 'targetVideoNums', [37 41 44 45 46]);
opts = setDefault(opts, 'includeLabels', {});              % {} => auto (all found)
opts = setDefault(opts, 'frameStride', 1);                 % 1 => all frames
opts = setDefault(opts, 'writeEmptyLabelFiles', false);    % YOLO can handle missing txt
opts = setDefault(opts, 'interpolateBoxes', true);         % interpolate between enabled keyframes
opts = setDefault(opts, 'cleanOutput', false);             % set true to wipe output folder first

baseDir = fileparts(mfilename('fullpath'));
exportJsonPath = fullfile(baseDir, char(exportJsonName));
videosDir      = fullfile(baseDir, char(opts.videosSubdir));
outDir         = fullfile(baseDir, char(opts.outDirName));

assert(isfile(exportJsonPath), "Export JSON not found: %s", exportJsonPath);
assert(isfolder(videosDir),    "Videos folder not found: %s", videosDir);

% Optional clean
if opts.cleanOutput && isfolder(outDir)
    rmdir(outDir, 's');
end
ensureDirs(outDir);

% Index actual videos in ./videos
videoMap = buildVideoMap(videosDir);

% Load JSON (array of tasks)
tasks = jsondecode(fileread(exportJsonPath));
assert(isstruct(tasks) || iscell(tasks), "Unexpected JSON format. Expected list of tasks.");

% Normalize tasks to struct array
if iscell(tasks), tasks = [tasks{:}]; end

% Filter tasks to only requested video numbers AND ones that exist locally
targetNums = unique(opts.targetVideoNums(:)');
keep = false(1, numel(tasks));
tailNames = strings(1, numel(tasks));
localPaths = strings(1, numel(tasks));

for i = 1:numel(tasks)
    if ~isfield(tasks(i),'file_upload') || isempty(tasks(i).file_upload), continue; end
    tail = stripHashPrefix(string(tasks(i).file_upload));    % "MAH00041.MP4"
    tailNames(i) = tail;

    num = parseMahNumber(tail);
    if ~ismember(num, targetNums), continue; end

    p = resolveVideoPath(videosDir, tail, videoMap);
    if strlength(p)==0
        fprintf("SKIP (not found in ./videos): %s\n", tail);
        continue
    end
    keep(i) = true;
    localPaths(i) = p;
end

tasks = tasks(keep);
tailNames = tailNames(keep);
localPaths = localPaths(keep);

assert(~isempty(tasks), "No matching tasks found for target videos in this export.");

% Split by VIDEO (prevents leakage)
vidList = unique(tailNames);
splits = splitVideos(vidList, opts.splitRatios, opts.seed);

videoToSplit = containers.Map('KeyType','char','ValueType','char');
for k = 1:numel(splits.train), videoToSplit(char(splits.train(k))) = 'train'; end
for k = 1:numel(splits.val),   videoToSplit(char(splits.val(k)))   = 'val';   end
for k = 1:numel(splits.test),  videoToSplit(char(splits.test(k)))  = 'test';  end

% Decide class list
if isempty(opts.includeLabels)
    classNames = collectLabels(tasks);
else
    classNames = string(opts.includeLabels(:));
end
classNames = unique(classNames, 'stable');
classId = containers.Map('KeyType','char','ValueType','int32');
for i = 1:numel(classNames)
    classId(char(classNames(i))) = int32(i-1);
end

fprintf("Base dir: %s\n", baseDir);
fprintf("Export:   %s\n", exportJsonPath);
fprintf("Videos:   %s\n", videosDir);
fprintf("Output:   %s\n", outDir);
fprintf("Target videos found: %d\n", numel(vidList));
fprintf("Split (by video): train=%d, val=%d, test=%d\n", numel(splits.train), numel(splits.val), numel(splits.test));
fprintf("Classes (%d):\n", numel(classNames)); disp(classNames);

totalImages = 0;

% ---- Process each task (one per video) ----
for ti = 1:numel(tasks)
    tail = stripHashPrefix(string(tasks(ti).file_upload));   % "MAH00045.MP4"
    vidPath = localPaths(ti);

    if isKey(videoToSplit, char(tail))
        splitName = videoToSplit(char(tail));
    else
        splitName = 'train';
    end

    % Extract framesCount from JSON (TRUST THIS, not VideoReader.Duration)
    framesCount = getFramesCount(tasks(ti));
    if isempty(framesCount) || framesCount < 1
        fprintf("SKIP (no framesCount): %s\n", tail);
        continue
    end

    % Build per-frame labels (1-based frames in your export)
    labelsPerFrame = cell(framesCount, 1);

    if ~isfield(tasks(ti),'annotations') || isempty(tasks(ti).annotations)
        fprintf("WARN: no annotations for %s (will export frames with no labels)\n", tail);
    else
        anns = tasks(ti).annotations;
        for ai = 1:numel(anns)
            if ~isfield(anns(ai),'result') || isempty(anns(ai).result), continue; end
            res = anns(ai).result;

            for ri = 1:numel(res)
                if ~isfield(res(ri),'type') || ~strcmp(res(ri).type, 'videorectangle'), continue; end
                if ~isfield(res(ri),'value'), continue; end

                val = res(ri).value;
                if ~isfield(val,'labels') || isempty(val.labels), continue; end
                lab = string(val.labels{1});
                if ~isKey(classId, char(lab)), continue; end

                if ~isfield(val,'sequence') || isempty(val.sequence), continue; end
                seq = val.sequence;

                labelsPerFrame = expandSequenceIntoFrames(labelsPerFrame, seq, ...
                    classId(char(lab)), framesCount, opts.interpolateBoxes);
            end
        end
    end

    % Output dirs
    imgOutDir = fullfile(outDir, 'images', splitName);
    labOutDir = fullfile(outDir, 'labels', splitName);

    if ~isfolder(imgOutDir), mkdir(imgOutDir); end
    if ~isfolder(labOutDir), mkdir(labOutDir); end


    % Write frames
    v = VideoReader(vidPath);
    frame0 = 0; % internal counter (0-based)
    wroteThisVideo = 0;

    [~, stem, ~] = fileparts(char(tail)); % e.g., "MAH00045"

    while hasFrame(v)
        frame0 = frame0 + 1;      % convert to LS frame index (1-based)
        if frame0 > framesCount
            break
        end

        img = readFrame(v);

        if mod(frame0-1, opts.frameStride) ~= 0
            continue
        end

        frameName = sprintf('%s_f%06d', stem, frame0); % store LS frame index
        imwrite(img, fullfile(imgOutDir, [frameName '.jpg']));

        lines = labelsPerFrame{frame0};
        if ~isempty(lines)
            writeLines(fullfile(labOutDir, [frameName '.txt']), lines);
        else
            if opts.writeEmptyLabelFiles
                fclose(fopen(fullfile(labOutDir, [frameName '.txt']), 'w'));
            end
        end

        wroteThisVideo = wroteThisVideo + 1;
    end

    totalImages = totalImages + wroteThisVideo;
    fprintf("[%s] framesCount=%d, wrote=%d (stride=%d) -> %s/%s\n", ...
        stem, framesCount, wroteThisVideo, opts.frameStride, splitName, stem);
end

% Write YAML
writeYaml(outDir, classNames);

fprintf("DONE. Wrote %d images total.\n", totalImages);
fprintf("YAML: %s\n", fullfile(outDir, 'dataset.yaml'));

end

% ================= Helpers =================

function framesCount = getFramesCount(task)
framesCount = [];
if ~isfield(task,'annotations') || isempty(task.annotations), return; end
anns = task.annotations;
fc = [];
for ai = 1:numel(anns)
    if ~isfield(anns(ai),'result') || isempty(anns(ai).result), continue; end
    res = anns(ai).result;
    for ri = 1:numel(res)
        if ~isfield(res(ri),'type') || ~strcmp(res(ri).type, 'videorectangle'), continue; end
        val = res(ri).value;
        if isfield(val,'framesCount') && ~isempty(val.framesCount)
            fc(end+1) = double(val.framesCount); %#ok<AGROW>
        end
    end
end
if ~isempty(fc)
    framesCount = max(fc);
end
end

function labelsPerFrame = expandSequenceIntoFrames(labelsPerFrame, seq, cid, framesCount, doInterp)
% seq is sparse "events": starting at seq(i).frame, state becomes enabled/disabled.
% We expand enabled intervals across frames until next event.
%
% IMPORTANT: Your export uses 1-based frames (min frame observed is 1).
% x,y,width,height are in percent (0-100), top-left anchored.

% Convert to struct array if needed
if iscell(seq), seq = [seq{:}]; end

% Keep only items with a frame
hasF = arrayfun(@(s) isfield(s,'frame') && ~isempty(s.frame), seq);
seq = seq(hasF);
if isempty(seq), return; end

% Sort by frame
frames = arrayfun(@(s) double(s.frame), seq);
[frames, order] = sort(frames);
seq = seq(order);

for i = 1:numel(seq)
    f1 = double(seq(i).frame);
    if f1 < 1 || f1 > framesCount, continue; end

    % Default enabled=true if missing
    en = true;
    if isfield(seq(i),'enabled') && ~isempty(seq(i).enabled)
        en = logical(seq(i).enabled);
    end
    if ~en
        continue
    end

    % Interval end is just before next event, else end of video
    if i < numel(seq)
        f2 = double(seq(i+1).frame) - 1;
    else
        f2 = framesCount;
    end
    f2 = min(max(f2, f1), framesCount);

    % Box at start
    if ~all(isfield(seq(i), {'x','y','width','height'})), continue; end
    x1 = double(seq(i).x); y1 = double(seq(i).y);
    w1 = double(seq(i).width); h1 = double(seq(i).height);

    % Box at end (optional interpolation if next is enabled==true)
    x2=x1; y2=y1; w2=w1; h2=h1;
    if doInterp && i < numel(seq) ...
            && isfield(seq(i+1),'enabled') && logical(seq(i+1).enabled) ...
            && all(isfield(seq(i+1), {'x','y','width','height'}))
        x2 = double(seq(i+1).x); y2 = double(seq(i+1).y);
        w2 = double(seq(i+1).width); h2 = double(seq(i+1).height);
    end

    span = f2 - f1;
    for f = f1:f2
        if span > 0
            t = (f - f1) / span;
        else
            t = 0;
        end
        x = x1 + t*(x2-x1);
        y = y1 + t*(y2-y1);
        w = w1 + t*(w2-w1);
        h = h1 + t*(h2-h1);

        line = yoloLine(cid, x, y, w, h);

        if isempty(labelsPerFrame{f})
            labelsPerFrame{f} = {line};
        else
            labelsPerFrame{f}{end+1} = line; %#ok<AGROW>
        end
    end
end
end

function s = stripHashPrefix(fileUpload)
% "cde13086-MAH00041.MP4" -> "MAH00041.MP4"
txt = char(fileUpload);
dash = find(txt=='-', 1, 'last');
if ~isempty(dash) && dash < numel(txt)
    s = string(txt(dash+1:end));
else
    s = string(txt);
end
end

function num = parseMahNumber(tail)
% "MAH00041.MP4" -> 41
num = NaN;
m = regexp(char(tail), 'MAH0*([0-9]+)\.MP4', 'tokens', 'once', 'ignorecase');
if ~isempty(m)
    num = str2double(m{1});
end
end

function classNames = collectLabels(tasks)
labs = strings(0,1);
for i = 1:numel(tasks)
    if ~isfield(tasks(i),'annotations') || isempty(tasks(i).annotations), continue; end
    anns = tasks(i).annotations;
    for ai = 1:numel(anns)
        if ~isfield(anns(ai),'result') || isempty(anns(ai).result), continue; end
        res = anns(ai).result;
        for ri = 1:numel(res)
            if ~isfield(res(ri),'type') || ~strcmp(res(ri).type, 'videorectangle'), continue; end
            val = res(ri).value;
            if isfield(val,'labels') && ~isempty(val.labels)
                labs(end+1,1) = string(val.labels{1}); %#ok<AGROW>
            end
        end
    end
end
classNames = unique(labs, 'stable');
end

function ensureDirs(outDir)
splits = {'train','val','test'};
for i = 1:numel(splits)
    mkdir(fullfile(outDir, 'images', splits{i}));
    mkdir(fullfile(outDir, 'labels', splits{i}));
end
end

function splits = splitVideos(vidList, ratios, seed)
N = numel(vidList);
rng(seed);
perm = randperm(N);
vidList = vidList(perm);

nTest = max(1, round(ratios(3) * N));
nVal  = max(1, round(ratios(2) * N));
nTrain = N - nVal - nTest;

% Make it sane for small N
if nTrain < 1
    nTrain = max(1, N-2);
    nVal = max(1, N-nTrain-1);
    nTest = N - nTrain - nVal;
end

splits.train = vidList(1:nTrain);
splits.val   = vidList(nTrain+1:nTrain+nVal);
splits.test  = vidList(nTrain+nVal+1:end);
end

function line = yoloLine(classId, xPct, yPct, wPct, hPct)
% Convert LS percent top-left box to YOLO normalized center box
x0 = xPct/100.0; y0 = yPct/100.0;
ww = wPct/100.0; hh = hPct/100.0;

% Clamp to [0,1]
x0 = min(max(x0, 0), 1); y0 = min(max(y0, 0), 1);
ww = min(max(ww, 0), 1); hh = min(max(hh, 0), 1);

cx = x0 + ww/2.0;
cy = y0 + hh/2.0;

cx = min(max(cx, 0), 1);
cy = min(max(cy, 0), 1);

line = sprintf('%d %.6f %.6f %.6f %.6f', classId, cx, cy, ww, hh);
end

function writeLines(path, lines)
% Robust writer that ensures folder exists and throws a useful error if fopen fails

outFolder = fileparts(path);
if ~isfolder(outFolder)
    mkdir(outFolder);
end

fid = fopen(path, 'w');
if fid < 0
    error("Could not open label file for writing:\n  %s\nTry a shorter outDirName or a local drive (C:), or check write permissions.", path);
end

for i = 1:numel(lines)
    % Ensure char for fprintf
    fprintf(fid, '%s\n', char(lines{i}));
end
fclose(fid);
end


function writeYaml(outDir, classNames)
yamlPath = fullfile(outDir, 'dataset.yaml');
fid = fopen(yamlPath,'w');
fprintf(fid, 'path: %s\n', outDir);
fprintf(fid, 'train: images/train\n');
fprintf(fid, 'val: images/val\n');
fprintf(fid, 'test: images/test\n');
fprintf(fid, 'names:\n');
for i = 1:numel(classNames)
    fprintf(fid, '  %d: %s\n', i-1, classNames(i));
end
fclose(fid);
end

function videoMap = buildVideoMap(videosDir)
files = [dir(fullfile(videosDir, '*.mp4')); dir(fullfile(videosDir, '*.MP4'))];
videoMap = containers.Map('KeyType','char','ValueType','char');
for i = 1:numel(files)
    videoMap(lower(files(i).name)) = fullfile(videosDir, files(i).name);
end
end

function vidPath = resolveVideoPath(videosDir, tailName, videoMap)
% tailName is like "MAH00041.MP4"
key = lower(char(tailName));
if isKey(videoMap, key)
    vidPath = string(videoMap(key));
else
    vidPath = "";
    % fallback: contains match
    files = [dir(fullfile(videosDir, '*.mp4')); dir(fullfile(videosDir, '*.MP4'))];
    for i = 1:numel(files)
        if contains(lower(files(i).name), lower(key))
            vidPath = string(fullfile(videosDir, files(i).name));
            return
        end
    end
end
end

function s = setDefault(s, field, value)
if ~isfield(s, field) || isempty(s.(field))
    s.(field) = value;
end
end
